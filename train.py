"""
Brain Tumor Detection - Comprehensive VGG16 Model Training
Uses Transfer Learning with fine-tuned convolutional layers on Brain MRI Scans.
Classifies into 4 classes: pituitary, glioma, notumor, meningioma.

Usage:
  python train.py --epochs 8 --batch_size 32
  python train.py --train_dir dataset/Training --test_dir dataset/Testing
"""

import os
import random
import argparse
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

IMAGE_SIZE = 128
CLASS_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']

def augment_image(img_arr):
    """Apply random flip, brightness, and contrast augmentation for MRI."""
    img = Image.fromarray(np.uint8(img_arr * 255.0))
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    return np.array(img, dtype=np.float32) / 255.0

def load_dataset(split_dir, samples_per_class=None, augment=False):
    """Load images and labels from class subdirectories."""
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
            if augment:
                arr = augment_image(arr)
            images.append(arr)
            labels.append(idx)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    return images[indices], labels[indices]

def build_model(learning_rate=0.0001):
    """Build VGG16 Transfer Learning model with fine-tuned block5."""
    base_model = VGG16(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze earlier layers
    for layer in base_model.layers:
        layer.trainable = False

    # Fine-tune the final convolutional block (block5) for MRI domain adaptation
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        base_model,
        Flatten(),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(len(CLASS_LABELS), activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="Train VGG16 Brain Tumor Detection Model")
    parser.add_argument("--train_dir", type=str, default="dataset/Training", help="Path to Training folder")
    parser.add_argument("--test_dir", type=str, default="dataset/Testing", help="Path to Testing folder")
    parser.add_argument("--samples_per_class", type=int, default=500, help="Samples per class (None for full)")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--output_model", type=str, default="models/model.h5", help="Path to save trained model")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    print("=" * 60)
    print(" Brain Tumor Detection - VGG16 Training")
    print("=" * 60)

    print(f"\n[1/4] Loading Training data from '{args.train_dir}'...")
    X_train, y_train = load_dataset(args.train_dir, samples_per_class=args.samples_per_class, augment=True)

    print(f"\n[2/4] Loading Validation data from '{args.test_dir}'...")
    val_samples = min(150, args.samples_per_class) if args.samples_per_class else None
    X_val, y_val = load_dataset(args.test_dir, samples_per_class=val_samples, augment=False)

    print(f"\n[3/4] Building model (learning_rate={args.lr})...")
    model = build_model(learning_rate=args.lr)
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1)
    ]

    print(f"\n[4/4] Training for up to {args.epochs} epochs (batch_size={args.batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"\n[INFO] Best Validation Accuracy: {val_acc * 100:.2f}%")

    print(f"[INFO] Saving model to '{args.output_model}'...")
    model.save(args.output_model)
    print(f"[SUCCESS] Model saved to '{args.output_model}'!")

if __name__ == "__main__":
    main()
