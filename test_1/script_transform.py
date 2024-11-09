import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

start_time = time.time()
print("Loading libraries ...")
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

library_load_time = time.time() - start_time
print(f"Library load time: {library_load_time:.4f} seconds")

def get_device():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" or device.type == "mps":
        print(f"Using device: {device} (GPU accelerated)")
    else:
        print(f"Using device: {device} (CPU based)")
    return device

def load_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    return preprocess(input_image).unsqueeze(0)

def predict_depth(model, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        depth = model(image_tensor)
    return depth.squeeze().cpu().numpy()

def save_depth_map(depth_map, output_path):
    plt.imsave(output_path, depth_map, cmap="gray")

print("Begin")

# Load models
model_load_start = time.time()
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS").eval()
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).eval()
device = get_device()
depth_model.to(device)
yolo_model.to(device)
model_load_time = time.time() - model_load_start
print(f"Model load time: {model_load_time:.4f} seconds")

# Load and preprocess image
image_path = "cat.png"
image_tensor = load_image(image_path)
image_pil = Image.open(image_path).convert("RGB")
image_np = np.array(image_pil)

# Predict depth
depth_map = predict_depth(depth_model, image_tensor)
save_depth_map(depth_map, "cat_depth_gray.png")
depth_map_normalized = (depth_map / depth_map.max() * 255).astype(np.uint8)

# Object detection with YOLO
results = yolo_model(image_np)

# Process detections and depth information
objects = []
for det in results.xyxy[0]:  # Cada detecció
    x1, y1, x2, y2, confidence, class_id = map(int, det[:6])
    width, height = x2 - x1, y2 - y1

    center_x = min((x1 + x2) // 2, depth_map_normalized.shape[1] - 1)
    center_y = min((y1 + y2) // 2, depth_map_normalized.shape[0] - 1)

    depth_value = depth_map_normalized[center_y, center_x]
    
    objects.append({
        "class": yolo_model.names[class_id],
        "position": {"x": center_x, "y": center_y, "z": depth_value},
        "size": {"width": width, "height": height}
    })

# Output results
for obj in objects:
    print(f"Object: {obj['class']}, Position - x: {obj['position']['x']}, y: {obj['position']['y']}, z (center depth): {obj['position']['z']}, Size - width: {obj['size']['width']}, height: {obj['size']['height']}")
