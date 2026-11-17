import os
import shutil
import numpy as np
from flask import Flask, render_template, request, send_from_directory, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Class definitions & clinical metadata
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
        "often characterized by diffuse infiltration into surrounding brain parenchyma. "
        "Clinical manifestations frequently include progressive headaches and focal neurological signs."
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

SAMPLE_IMAGES = {
    'glioma': 'Te-gl_0015.jpg',
    'meningioma': 'Te-meTr_0001.jpg',
    'notumor': 'Te-noTr_0004.jpg',
    'pituitary': 'Te-piTr_0003.jpg'
}

# Directories
MODEL_PATH = os.path.join('models', 'model.h5')
UPLOAD_FOLDER = os.path.join('.', 'uploads')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global model instance
model = None

def get_model():
    """Lazily load the trained Keras/TensorFlow model."""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            from tensorflow.keras.models import load_model
            try:
                model = load_model(MODEL_PATH)
                print(f"[INFO] Successfully loaded model from '{MODEL_PATH}'")
            except Exception as e:
                print(f"[ERROR] Failed to load model from '{MODEL_PATH}': {e}")
                raise e
        else:
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Please ensure the model has finished training or place 'model.h5' inside 'models/'."
            )
    return model

def predict_tumor(image_path):
    """Preprocess MRI image and return detailed prediction breakdown."""
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    loaded_model = get_model()
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = loaded_model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    top_label = CLASS_LABELS[pred_idx]
    confidence = float(predictions[pred_idx])

    probabilities = []
    for idx, label in enumerate(CLASS_LABELS):
        pct = float(predictions[idx]) * 100.0
        probabilities.append({
            'key': label,
            'name': CLASS_DISPLAY[label],
            'percentage': f"{pct:.2f}%",
            'value': round(pct, 2),
            'color': CLASS_COLORS[label],
            'is_top': (idx == pred_idx)
        })

    # Sort probabilities descending for clean chart display
    probabilities = sorted(probabilities, key=lambda x: x['value'], reverse=True)

    result_class = CLASS_DISPLAY[top_label]
    confidence_formatted = f"{confidence * 100:.2f}%"

    return {
        'top_key': top_label,
        'result': result_class,
        'confidence': confidence_formatted,
        'confidence_val': confidence * 100,
        'color': CLASS_COLORS[top_label],
        'description': CLASS_DESCRIPTIONS[top_label],
        'probabilities': probabilities
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            try:
                prediction_data = predict_tumor(file_location)
                return render_template(
                    'index.html',
                    result=prediction_data['result'],
                    confidence=prediction_data['confidence'],
                    file_path=f"/uploads/{file.filename}",
                    prediction=prediction_data,
                    probabilities=prediction_data['probabilities']
                )
            except FileNotFoundError as fnf_err:
                return render_template(
                    'index.html',
                    error=str(fnf_err),
                    file_path=f"/uploads/{file.filename}"
                )
            except Exception as e:
                return render_template(
                    'index.html',
                    error=f"Prediction error: {str(e)}",
                    file_path=f"/uploads/{file.filename}"
                )

    warning = None
    if not os.path.exists(MODEL_PATH):
        warning = "Model file 'models/model.h5' not found. Training may still be in progress."

    return render_template(
        'index.html',
        result=None,
        confidence=None,
        file_path=None,
        prediction=None,
        probabilities=None,
        warning=warning
    )

@app.route('/sample/<tumor_type>')
def test_sample(tumor_type):
    """Convenience endpoint to test with root sample images with 1 click."""
    if tumor_type in SAMPLE_IMAGES:
        src = SAMPLE_IMAGES[tumor_type]
        if os.path.exists(src):
            dest = os.path.join(app.config['UPLOAD_FOLDER'], f"sample_{src}")
            shutil.copyfile(src, dest)
            try:
                prediction_data = predict_tumor(dest)
                return render_template(
                    'index.html',
                    result=prediction_data['result'],
                    confidence=prediction_data['confidence'],
                    file_path=f"/uploads/sample_{src}",
                    prediction=prediction_data,
                    probabilities=prediction_data['probabilities']
                )
            except Exception as e:
                return render_template('index.html', error=str(e))
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("[INFO] Starting NeuroScan AI Web Server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
