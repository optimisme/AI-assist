import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
print("Loading libraries ...")
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from flask import Flask, request, jsonify

SERVER_PORT = 8400

app = Flask(__name__)

def get_device():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" or device.type == "mps":
        print(f"Using device: {device} (GPU accelerated)")
    else:
        print(f"Using device: {device} (CPU based)")
    return device

def load_image_from_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    return preprocess(image).unsqueeze(0).to(device), np.array(image)

def predict_depth(model, image_tensor):
    with torch.no_grad():
        depth = model(image_tensor)
    return depth.squeeze().cpu().numpy()

def depth_map_to_base64(depth_map):
    buffer = io.BytesIO()
    plt.imsave(buffer, depth_map, cmap="gray", format="png")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def detect_objects(yolo_model, image_np, depth_map_normalized):
    results = yolo_model(image_np)
    objects = []
    for det in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = map(int, det[:6])
        width, height = x2 - x1, y2 - y1
        center_x = min((x1 + x2) // 2, depth_map_normalized.shape[1] - 1)
        center_y = min((y1 + y2) // 2, depth_map_normalized.shape[0] - 1)
        depth_value = depth_map_normalized[center_y, center_x]
        objects.append({
            "class": yolo_model.names[class_id],
            "position": {"x": center_x, "y": center_y, "z": int(depth_value)},
            "size": {"width": width, "height": height}
        })
    return objects

@app.route('/depth_and_objects', methods=['POST'])
def depth_and_objects_inference():
    data = request.get_json()
    base64_image = data.get("image")
    
    start_time = time.time()
    image_tensor, image_np = load_image_from_base64(base64_image)
    load_time = time.time() - start_time

    start_time = time.time()
    depth_map = predict_depth(depth_model, image_tensor)
    depth_map_normalized = (depth_map / depth_map.max() * 255).astype(np.uint8)
    inference_time = time.time() - start_time

    start_time = time.time()
    result_base64 = depth_map_to_base64(depth_map_normalized)
    objects = detect_objects(yolo_model, image_np, depth_map_normalized)
    processing_time = time.time() - start_time

    return jsonify({
        "depth_image": result_base64,
        "objects": objects,
        "times": {
            "load_time": load_time,
            "inference_time": inference_time,
            "processing_time": processing_time
        }
    })

if __name__ == "__main__":
    depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS").eval()
    yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).eval()
    device = get_device()
    depth_model.to(device)
    yolo_model.to(device)
    app.run(host="0.0.0.0", port=SERVER_PORT)
