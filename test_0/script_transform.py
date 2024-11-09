import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Measure library load time
start_time = time.time()
print("Loading libraries ...")
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
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

def predict_depth(model, image_tensor, warmup=False):
    image_tensor = image_tensor.to(device)
    if warmup:
        warmup_start = time.time()
        with torch.no_grad():
            _ = model(image_tensor)  # Warm-up pass to load model on GPU
        warmup_time = time.time() - warmup_start
        print(f"Warm-up (GPU load) time: {warmup_time:.4f} seconds")
        return
    
    start_time = time.time()
    with torch.no_grad():
        depth = model(image_tensor)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")
    return depth.squeeze().cpu().numpy()

def save_depth_map(depth_map, output_path):
    plt.imsave(output_path, depth_map, cmap="gray")

print("Begin")

# Measure model load time
model_load_start = time.time()
model = torch.hub.load("intel-isl/MiDaS", "MiDaS").eval()
device = get_device()
model.to(device)
model_load_time = time.time() - model_load_start
print(f"Model load time: {model_load_time:.4f} seconds")

# Load and preprocess image
image_tensor = load_image("cat.png")

# Warm-up pass
predict_depth(model, image_tensor, warmup=True)

# Actual inference with time measurement
depth_map = predict_depth(model, image_tensor)
save_depth_map(depth_map, "cat_depth_gray.png")
