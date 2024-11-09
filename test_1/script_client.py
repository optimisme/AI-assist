import base64
import requests
from PIL import Image
import io

SERVER_PORT = 8400

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def save_base64_image_to_file_direct(base64_str, output_path):
    image_data = base64.b64decode(base64_str)
    with Image.open(io.BytesIO(image_data)) as image:
        image.save(output_path)

image_base64 = encode_image_to_base64("stairs.png")

url = f"http://127.0.0.1:{SERVER_PORT}/depth_and_objects"

print("Call begin")
response = requests.post(url, json={"image": image_base64})

if response.status_code == 200:
    response_data = response.json()
    
    times = response_data["times"]
    print("Processing times:")
    print(f"  Load time: {times['load_time']:.3f} seconds")
    print(f"  Inference time: {times['inference_time']:.3f} seconds")
    print(f"  Processing time: {times['processing_time']:.3f} seconds")
    
    depth_image_base64 = response_data["depth_image"]
    out_path = "stairs_depth.png"
    save_base64_image_to_file_direct(depth_image_base64, out_path)
    print(f"Depth map saved as {out_path}")
    
    objects = response_data.get("objects", [])
    if objects:
        print("Detected objects:")
        for obj in objects:
            print(f"  Object: {obj['class']}")
            print(f"    Position - x: {obj['position']['x']}, y: {obj['position']['y']}, z (depth): {obj['position']['z']}")
            print(f"    Size - width: {obj['size']['width']}, height: {obj['size']['height']}")
    else:
        print("No objects detected.")
else:
    print("Failed to get response from the server:", response.status_code, response.text)
