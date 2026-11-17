"""
prediction_service.py - NeuroScan AI Prediction Engine
Handles image loading, preprocessing, model inference, and metadata extraction.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = os.path.join('models', 'model.h5')
IMAGE_SIZE = 128

CLASS_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']

CLASS_DISPLAY = {
    'pituitary': 'Pituitary',
    'glioma': 'Glioma',
    'notumor': 'No Tumor',
    'meningioma': 'Meningioma'
}

CLASS_DESCRIPTIONS = {
    'glioma': (
        "Gliomas originate in the brain's glial support cells. They are intra-axial neoplasms "
        "characterized by diffuse infiltration into surrounding brain parenchyma. "
        "Clinical presentations frequently include progressive headaches and focal neurological signs."
    ),
    'meningioma': (
        "Meningiomas arise from arachnoid cap cells of the meninges covering the brain. "
        "The vast majority are slow-growing extra-axial tumors that primarily produce symptoms "
        "via mechanical compression of adjacent brain tissue and cranial nerves."
    ),
    'pituitary': (
        "Pituitary adenomas develop in the sella turcica at the skull base. "
        "They commonly present with neuroendocrine hormone alterations or bitemporal hemianopsia "
        "due to superior compression of the optic chiasm."
    ),
    'notumor': (
        "Normal physiological brain MRI scan. Brain parenchyma, ventricular contours, "
        "and midline structures demonstrate normal anatomical integrity without evidence of space-occupying lesions."
    )
}

CLASS_COLORS = {
    'glioma': 'danger',
    'meningioma': 'warning',
    'pituitary': 'info',
    'notumor': 'success'
}

# Hex colors for Chart.js
CLASS_HEX_COLORS = {
    'glioma': '#dc2626',
    'meningioma': '#d97706',
    'pituitary': '#7c3aed',
    'notumor': '#16a34a'
}

_model = None

def get_model():
    """Lazily load and cache the trained VGG16 model."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Ensure training has finished or place 'model.h5' inside 'models/'."
            )
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
        print(f"[INFO] Loaded model from '{MODEL_PATH}'")
    return _model

def get_image_metadata(image_path):
    """Extract format, dimensions, and file size for clinical metadata display."""
    stat = os.stat(image_path)
    size_bytes = stat.st_size
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"

    with Image.open(image_path) as im:
        fmt = im.format or "JPEG"
        width, height = im.size

    return {
        'filename': os.path.basename(image_path),
        'dimensions': f"{width} × {height}",
        'width': width,
        'height': height,
        'format': fmt,
        'size': size_str
    }

def predict_mri(image_path):
    """
    Execute full inference pipeline:
    1. Preprocess 128x128 normalized tensor
    2. Extract model softmax probabilities
    3. Return structured prediction payload
    """
    model = get_model()

    # Model input preprocessing
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    tensor = np.expand_dims(img_array, axis=0)

    # Inference
    predictions = model.predict(tensor, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    top_label = CLASS_LABELS[pred_idx]
    confidence = float(predictions[pred_idx])

    # Probabilities breakdown for display
    probabilities = []
    chart_labels = []
    chart_values = []
    chart_colors = []

    for idx, label in enumerate(CLASS_LABELS):
        pct = float(predictions[idx]) * 100.0
        probabilities.append({
            'key': label,
            'name': CLASS_DISPLAY[label],
            'percentage': f"{pct:.2f}%",
            'value': round(pct, 2),
            'color': CLASS_COLORS[label],
            'hex_color': CLASS_HEX_COLORS[label],
            'is_top': (idx == pred_idx)
        })
        chart_labels.append(CLASS_DISPLAY[label])
        chart_values.append(round(pct, 2))
        chart_colors.append(CLASS_HEX_COLORS[label])

    # Sort display list descending
    probabilities_sorted = sorted(probabilities, key=lambda x: x['value'], reverse=True)

    metadata = get_image_metadata(image_path)

    return {
        'top_key': top_label,
        'result': CLASS_DISPLAY[top_label],
        'confidence': f"{confidence * 100:.2f}%",
        'confidence_val': round(confidence * 100, 2),
        'color': CLASS_COLORS[top_label],
        'hex_color': CLASS_HEX_COLORS[top_label],
        'description': CLASS_DESCRIPTIONS[top_label],
        'probabilities': probabilities_sorted,
        'chart_labels': chart_labels,
        'chart_values': chart_values,
        'chart_colors': chart_colors,
        'metadata': metadata
    }
