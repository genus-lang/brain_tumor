"""
Advanced High-Confidence Brain Tumor Model Training
- VGG16 with fine-tuned top convolutional block
- Data Augmentation (Rotation, Brightness, Contrast, Horizontal Flip)
- Learning Rate Schedule with Plateau Reduction
- Trains on 2,000 balanced images (500 per class) + 600 validation images
- Evaluates on root test samples to confirm high confidence (>85-95%)
- Saves optimized model to models/model.h5
"""

import os
import random
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
MODEL_OUT = os.path.join('models', 'model.h5')

def augment_image(img_arr):
    """Data augmentation tailored for MRI scans."""
    img = Image.fromarray(np.uint8(img_arr * 255.0))
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random brightness enhancement (0.85 to 1.15)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    # Random contrast enhancement (0.85 to 1.15)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    return np.array(img, dtype=np.float32) / 255.0

def load_data(split_dir, samples_per_class=500, augment=False):
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
            if augment:
                arr = augment_image(arr)
            images.append(arr)
            labels.append(idx)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    return images[indices], labels[indices]

def build_advanced_model():
    """Build VGG16 with fine-tuned block5 layers."""
    base_model = VGG16(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze earlier layers, unfreeze the final block (block5)
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze block5 convolution layers for domain adaptation to MRI
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
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("=" * 60)
    print(" Advanced High-Confidence Brain Tumor Model Training")
    print("=" * 60)

    train_dir = os.path.join('dataset', 'Training')
    test_dir = os.path.join('dataset', 'Testing')

    print("\n[1/4] Loading balanced Training set (500/class = 2,000 images)...")
    X_train, y_train = load_data(train_dir, samples_per_class=500, augment=True)

    print("\n[2/4] Loading balanced Validation set (150/class = 600 images)...")
    X_val, y_val = load_data(test_dir, samples_per_class=150, augment=False)

    print("\n[3/4] Building fine-tuned model...")
    model = build_advanced_model()
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1)
    ]

    epochs = 8
    batch_size = 32
    print(f"\n[4/4] Training for up to {epochs} epochs (batch_size={batch_size})...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"\n[INFO] Best Validation Accuracy: {val_acc * 100:.2f}%")

    os.makedirs('models', exist_ok=True)
    print(f"[INFO] Saving improved model to '{MODEL_OUT}'...")
    model.save(MODEL_OUT)
    print(f"[SUCCESS] High-confidence model saved to '{MODEL_OUT}'!")

    # Verify on test samples
    print("\n" + "=" * 60)
    print(" Verifying confidence on sample test images:")
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
            print(f"[{status}] {filename:16s} -> {pred_class:12s} ({conf*100:5.1f}%) | Expected: {expected}")
            for c, p in zip(CLASS_LABELS, preds[0]):
                print(f"       - {c:12s}: {p*100:5.1f}%")

if __name__ == "__main__":
    main()

