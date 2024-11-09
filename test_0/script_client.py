import base64
import requests
from PIL import Image
import io

SERVER_PORT = 8400

# Carregar la imatge i convertir-la a base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Decodificar la imatge de profunditat en base64 i guardar-la directament sense arxiu temporal
def save_base64_image_to_file_direct(base64_str, output_path):
    image_data = base64.b64decode(base64_str)
    with Image.open(io.BytesIO(image_data)) as image:
        image.save(output_path)

# Carregar la imatge
image_base64 = encode_image_to_base64("stairs.png")

# Configurar la URL del servidor (ajusta-ho si el servidor està en un altre host o port)
url = f"http://127.0.0.1:{SERVER_PORT}/depth"

# Enviar la sol·licitud
print("Call begin")
response = requests.post(url, json={"image": image_base64})

# Processar la resposta
if response.status_code == 200:
    response_data = response.json()
    
    # Imprimir els temps de processament
    times = response_data["times"]
    print("Processing times:")
    print(f"  Load time: {times['load_time']:.3f} seconds")
    print(f"  Inference time: {times['inference_time']:.3f} seconds")
    print(f"  Processing time: {times['processing_time']:.3f} seconds")
    
    # Desar la imatge de profunditat
    depth_image_base64 = response_data["depth_image"]
    out_path = "stairs_depth.png"
    save_base64_image_to_file_direct(depth_image_base64, out_path)
    print(f"Depth map saved as {out_path}")
else:
    print("Failed to get response from the server:", response.status_code, response.text)
