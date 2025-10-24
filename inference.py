import cv2
import numpy as np
from openvino.runtime import Core
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MODEL_PATH = "casting_ir/model.xml"
THRESHOLD = 0.0004  
IMG_SIZE = 304

# --- Initialize OpenVINO ---
ie = Core()
model = ie.read_model(model=MODEL_PATH)
compiled_model = ie.compile_model(model=model, device_name="CPU")
infer_request = compiled_model.create_infer_request()

# --- Preprocessing ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = np.stack([img]*3, axis=0)  # Shape: [3, 304, 304]
    img = np.expand_dims(img, 0)     # Shape: [1, 3, 304, 304]
    return img

# --- Reconstruction error ---
def reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed)**2)

# --- Save reconstruction ---
def save_reconstruction(img, reconstructed, output_path="reconstruction.png"):
    original = img[0].transpose(1,2,0)
    recon = reconstructed[0].transpose(1,2,0)
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.subplot(1,2,2)
    plt.title("Reconstructed")
    plt.imshow(recon, cmap='gray')
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Reconstruction saved to {output_path}")

# --- Detect anomaly ---
def detect_anomaly(image_path, threshold=THRESHOLD):
    img = preprocess_image(image_path)
    result = infer_request.infer(inputs={compiled_model.inputs[0]: img})
    reconstructed = result[compiled_model.outputs[0]]
    
    error = reconstruction_error(img, reconstructed)
    print(f"[INFO] Reconstruction error: {error:.6f}")
    
    save_reconstruction(img, reconstructed)
    
    if error > threshold:
        print("Defective Casting Detected ✅")
    else:
        print("Casting OK ✅")

# --- Run ---
if __name__ == "__main__":
    test_image_path = "cast_def_0_174.jpeg"  # replace with your image
    detect_anomaly(test_image_path)

