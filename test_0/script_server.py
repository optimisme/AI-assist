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
    return preprocess(image).unsqueeze(0).to(device)

def predict_depth(model, image_tensor):
    with torch.no_grad():
        depth = model(image_tensor)
    return depth.squeeze().cpu().numpy()

def depth_map_to_base64(depth_map):
    buffer = io.BytesIO()
    plt.imsave(buffer, depth_map, cmap="gray", format="png")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

@app.route('/depth', methods=['POST'])
def depth_inference():
    data = request.get_json()
    base64_image = data.get("image")

    # Processar la imatge de la petició
    start_time = time.time()
    image_tensor = load_image_from_base64(base64_image)
    load_time = time.time() - start_time

    # Inferència del model
    start_time = time.time()
    depth_map = predict_depth(model, image_tensor)
    inference_time = time.time() - start_time

    # Convertir el depth map en base64 directament sense arxiu temporal
    start_time = time.time()
    result_base64 = depth_map_to_base64(depth_map)
    processing_time = time.time() - start_time

    return jsonify({
        "depth_image": result_base64,
        "times": {
            "load_time": load_time,
            "inference_time": inference_time,
            "processing_time": processing_time
        }
    })

if __name__ == "__main__":
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS").eval()
    device = get_device()
    model.to(device)

    app.run(host="0.0.0.0", port=SERVER_PORT)
