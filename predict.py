"""
CLI Image Inference Script for Brain Tumor Detection
Usage: python predict.py <path_to_image> [--model_path models/model.h5]
"""

import sys
import os
import argparse
import numpy as np

CLASS_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']

def predict(image_path, model_path="models/model.h5"):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file '{model_path}' not found.")
        print("Please train the model or place 'model.h5' inside 'models/' directory.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"[ERROR] Image file '{image_path}' not found.")
        sys.exit(1)

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    print(f"[INFO] Loading model from '{model_path}'...")
    model = load_model(model_path)

    IMAGE_SIZE = 128
    print(f"[INFO] Processing image '{image_path}'...")
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions, axis=1)[0])

    label = CLASS_LABELS[class_idx]
    result = "No Tumor" if label == "notumor" else f"Tumor: {label}"

    print("\n" + "=" * 45)
    print(f" Diagnosis : {result}")
    print(f" Confidence: {confidence * 100:.2f}%")
    print(" Class Probabilities:")
    for cls, prob in zip(CLASS_LABELS, predictions[0]):
        print(f"   - {cls:12s}: {prob * 100:6.2f}%")
    print("=" * 45 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Predict brain tumor from MRI image")
    parser.add_argument("image", type=str, help="Path to MRI image file (.jpg, .png)")
    parser.add_argument("--model", type=str, default="models/model.h5", help="Path to trained model (.h5)")
    args = parser.parse_args()

    predict(args.image, args.model)

if __name__ == "__main__":
    main()

