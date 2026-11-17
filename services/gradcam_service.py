"""
gradcam_service.py - NeuroScan AI Explainable AI (XAI) Engine
Computes mathematical Grad-CAM activations from the trained VGG16 backbone,
applies anatomical brain masking, soft thresholding, and JET colormapping.
"""

import os
import uuid
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from services.prediction_service import get_model, IMAGE_SIZE

RESULTS_FOLDER = os.path.join('static', 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

_grad_model = None

def get_grad_model():
    """
    Constructs and caches a functional Gradient Model connecting the input
    to both the final convolutional feature maps (block5_conv3) and the final classification output.
    """
    global _grad_model
    if _grad_model is None:
        base_model = get_model()
        
        # Access the nested VGG16 functional backbone
        vgg = base_model.get_layer('vgg16')
        
        # Dynamically locate the final Conv2D layer in VGG16 (block5_conv3)
        last_conv = None
        for layer in reversed(vgg.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break
                
        if last_conv is None:
            raise ValueError("No Conv2D layer found inside VGG16 backbone for Grad-CAM.")
            
        print(f"[INFO] Grad-CAM targeting final convolutional layer: '{last_conv.name}'")

        # Build submodel that yields [last_conv_output, vgg_final_pool_output]
        vgg_submodel = tf.keras.Model(
            inputs=vgg.inputs,
            outputs=[last_conv.output, vgg.output]
        )

        # Build full end-to-end functional model
        inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        conv_out, vgg_out = vgg_submodel(inputs)

        # Propagate through sequential classification head
        x = vgg_out
        for layer in base_model.layers[1:]:
            x = layer(x)

        _grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_out, x])
        print("[INFO] Successfully built and cached Grad-CAM computation model.")

    return _grad_model

def generate_gradcam(image_path, target_class_idx=None):
    """
    Generate authentic Grad-CAM heatmap and overlay for the given MRI image.
    Applies anatomical brain masking to eliminate background air artifacts and
    soft thresholding to focus cleanly on pathological attention zones.
    
    Returns:
        dict: {
            'heatmap_url': str,
            'overlay_url': str,
            'original_url': str,
            'target_layer': str,
            'target_class_idx': int
        }
    """
    grad_model = get_grad_model()

    # 1. Preprocess 128x128 tensor
    img_tensor = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img_tensor) / 255.0
    tensor = np.expand_dims(img_array, axis=0)

    # 2. Gradient Tape for Grad-CAM calculation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tensor)
        if target_class_idx is None:
            target_class_idx = int(tf.argmax(predictions[0]))
        
        # Log-probability prevents vanishing gradients when confidence is saturated (> 99%)
        loss = tf.math.log(predictions[:, target_class_idx] + 1e-12)

    # Gradients of class score with respect to feature map activations
    grads = tape.gradient(loss, conv_outputs)

    # Channel-wise global average pooling (alpha weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    # ReLU to keep only positive contributions to target class
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize raw heatmap
    max_val = float(tf.reduce_max(heatmap))
    if max_val > 0:
        heatmap = (heatmap / max_val).numpy()
    else:
        heatmap = heatmap.numpy()

    # 3. Load high-resolution original image
    orig_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = orig_img.size

    # Bilinear resize heatmap to original dimensions
    heatmap_pil = Image.fromarray(np.uint8(heatmap * 255.0))
    heatmap_resized = heatmap_pil.resize((orig_w, orig_h), resample=Image.BILINEAR)
    heatmap_norm = np.array(heatmap_resized, dtype=np.float32) / 255.0

    # 4. Extract anatomical brain parenchyma mask
    # This strips the black background air and prevents edge padding artifacts from glowing
    orig_gray = orig_img.convert('L')
    orig_smooth = orig_gray.filter(ImageFilter.GaussianBlur(radius=2))
    brain_mask = binary_dilation(np.array(orig_smooth) > 15, iterations=4).astype(np.float32)

    # Apply brain mask to confine heatmap to brain tissue
    heatmap_masked = heatmap_norm * brain_mask
    max_brain = float(heatmap_masked.max())
    if max_brain > 0:
        heatmap_masked = heatmap_masked / max_brain

    # 5. Soft-threshold background noise (< 0.16)
    heatmap_thresh = np.clip((heatmap_masked - 0.16) / 0.84, 0.0, 1.0)

    # 6. Apply clinical JET colormap
    colormap = plt.get_cmap('jet')
    colored_heatmap = colormap(heatmap_thresh)[:, :, :3]
    
    # Pure heatmap: background and neutral areas become crisp PURE BLACK (0, 0, 0)
    # Only active attention zones glow vibrantly through the thermal spectrum
    black_bg = np.zeros_like(colored_heatmap)
    pure_heatmap_rgb = np.where(heatmap_thresh[:, :, None] > 0.02, colored_heatmap, black_bg)
    pure_heatmap_uint8 = np.uint8(pure_heatmap_rgb * 255.0)

    # 7. Superimpose onto original MRI scan with dynamic alpha blending
    # Healthy brain tissue retains its sharp grayscale MRI anatomy; tumor glows with thermal overlay
    orig_arr = np.array(orig_img, dtype=np.float32)
    alpha_map = (heatmap_thresh * 0.52)[:, :, None]
    overlay_arr = np.uint8(orig_arr * (1.0 - alpha_map) + (pure_heatmap_rgb * 255.0) * alpha_map)

    # 8. Save output files
    file_id = uuid.uuid4().hex[:10]
    heatmap_filename = f"heatmap_{file_id}.jpg"
    overlay_filename = f"overlay_{file_id}.jpg"

    heatmap_path = os.path.join(RESULTS_FOLDER, heatmap_filename)
    overlay_path = os.path.join(RESULTS_FOLDER, overlay_filename)

    Image.fromarray(pure_heatmap_uint8).save(heatmap_path, quality=95)
    Image.fromarray(overlay_arr).save(overlay_path, quality=95)

    return {
        'heatmap_url': f"/static/results/{heatmap_filename}",
        'overlay_url': f"/static/results/{overlay_filename}",
        'target_layer': 'block5_conv3',
        'target_class_idx': target_class_idx
    }

