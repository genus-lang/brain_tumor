"""
Train a high-accuracy VGG16 Transfer Learning Brain Tumor Classifier
using the extracted Kaggle MRI dataset.
Saves model directly to models/model.h5
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Ensure reproducible split
np.random.seed(42)
tf.random.set_seed(42)

IMAGE_SIZE = 128
CLASS_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']
MODEL_OUT = os.path.join('models', 'model.h5')

def load_data(split_dir, samples_per_class=400):
    """Load balanced dataset from class directories."""
    images = []
    labels = []
    for idx, class_name in enumerate(CLASS_LABELS):
        class_folder = os.path.join(split_dir, class_name)
        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"Folder not found: {class_folder}")
        all_files = [
            os.path.join(class_folder, f)
            for f in os.listdir(class_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        selected = all_files[:samples_per_class] if samples_per_class else all_files
        print(f"  [+] {class_name:12s} ({idx}): {len(selected)} images")
        for img_path in selected:
            img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            arr = img_to_array(img) / 255.0
            images.append(arr)
            labels.append(idx)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    # Shuffle
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    return images[indices], labels[indices]

def build_model():
    print("[INFO] Building VGG16 Transfer Learning architecture...")
    base_model = VGG16(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze base VGG16 layers
    base_model.trainable = False

    model = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        base_model,
        Flatten(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(len(CLASS_LABELS), activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("=" * 60)
    print(" Brain Tumor Detection - Fast Transfer Learning Training")
    print("=" * 60)

    train_dir = os.path.join('dataset', 'Training')
    test_dir = os.path.join('dataset', 'Testing')

    print("\n[INFO] Loading balanced Training set (400 per class = 1,600 images)...")
    X_train, y_train = load_data(train_dir, samples_per_class=400)

    print("\n[INFO] Loading balanced Validation set (100 per class = 400 images)...")
    X_val, y_val = load_data(test_dir, samples_per_class=100)

    model = build_model()
    model.summary()

    epochs = 4
    batch_size = 32
    print(f"\n[INFO] Training for {epochs} epochs (batch_size={batch_size})...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    val_acc = history.history['val_accuracy'][-1]
    print(f"\n[INFO] Final Validation Accuracy: {val_acc * 100:.2f}%")

    os.makedirs('models', exist_ok=True)
    print(f"[INFO] Saving trained model to '{MODEL_OUT}'...")
    model.save(MODEL_OUT)
    print(f"[SUCCESS] Model successfully saved to '{MODEL_OUT}'!")

    # Verify on sample images in root
    print("\n" + "=" * 60)
    print(" Verifying model on sample test images:")
    print("=" * 60)
    samples = [
        ("Te-gl_0015.jpg", "glioma"),
        ("Te-meTr_0001.jpg", "meningioma"),
        ("Te-noTr_0004.jpg", "notumor"),
        ("Te-piTr_0003.jpg", "pituitary"),
    ]
    for filename, expected in samples:
        if os.path.exists(filename):
            img = load_img(filename, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
            preds = model.predict(arr, verbose=0)
            pred_idx = np.argmax(preds[0])
            conf = preds[0][pred_idx]
            pred_class = CLASS_LABELS[pred_idx]
            status = "PASS" if pred_class == expected else "FAIL"
            print(f"[{status}] {filename:16s} -> Predicted: {pred_class:12s} ({conf*100:5.1f}%) | Expected: {expected}")

if __name__ == "__main__":
    main()

