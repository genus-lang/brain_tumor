"""
Brain Tumor Detection - Model Training Script
Uses Transfer Learning with VGG16 on Brain MRI Scans.
Classifies into 4 classes: glioma, meningioma, notumor, pituitary.
"""

import os
import random
import argparse
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.utils import shuffle

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img

IMAGE_SIZE = 128

def augment_image(image):
    """Apply random brightness and contrast augmentation, normalize to [0, 1]."""
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    return np.array(image) / 255.0

def open_images(paths):
    """Load and augment a batch of image paths."""
    images = []
    for path in paths:
        img = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = augment_image(img)
        images.append(img)
    return np.array(images)

def encode_labels(labels, unique_labels):
    """Convert string class labels to integer IDs."""
    return np.array([unique_labels.index(label) for label in labels])

def datagen(paths, labels, unique_labels, batch_size=20, epochs=5):
    """Yield batches of augmented images and encoded labels."""
    for _ in range(epochs):
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_images = open_images(batch_paths)
            batch_labels = encode_labels(labels[i:i + batch_size], unique_labels)
            yield batch_images, batch_labels

def build_model(num_classes):
    """Build VGG16 Transfer Learning model."""
    base_model = VGG16(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the last block for fine-tuning
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        base_model,
        Flatten(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    return model

def load_dataset(data_dir):
    """Load file paths and labels from a directory structured as class folders."""
    paths = []
    labels = []
    classes = sorted(os.listdir(data_dir))
    for label in classes:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(label_dir, filename))
                    labels.append(label)

    return shuffle(paths, labels, random_state=42)

def main():
    parser = argparse.ArgumentParser(description="Train VGG16 Brain Tumor Detection Model")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to Training folder")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to Testing folder (optional)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--output_model", type=str, default="models/model.h5", help="Path to save trained model")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    print(f"[INFO] Loading training data from: {args.train_dir}")
    train_paths, train_labels = load_dataset(args.train_dir)
    unique_labels = sorted(list(set(train_labels)))
    print(f"[INFO] Found {len(train_paths)} images across {len(unique_labels)} classes: {unique_labels}")

    model = build_model(len(unique_labels))
    model.summary()

    steps = int(len(train_paths) / args.batch_size)
    print(f"[INFO] Starting training for {args.epochs} epochs ({steps} steps per epoch)...")

    model.fit(
        datagen(train_paths, train_labels, unique_labels, batch_size=args.batch_size, epochs=args.epochs),
        epochs=args.epochs,
        steps_per_epoch=steps
    )

    print(f"[INFO] Saving trained model to {args.output_model}...")
    model.save(args.output_model)
    print("[SUCCESS] Training completed and model saved successfully!")

if __name__ == "__main__":
    main()

