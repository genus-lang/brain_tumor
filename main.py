"""
main.py - NeuroScan AI Web Application
Modular Flask backend powering AI Brain MRI Analysis, Grad-CAM XAI,
Model Benchmarking, and the AI Brain Tumor Care Assistant.
"""

import os
import shutil
import json
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify

from services.prediction_service import (
    predict_mri,
    CLASS_LABELS,
    CLASS_DISPLAY,
    MODEL_PATH,
    SAMPLE_IMAGES
)
from services.gradcam_service import generate_gradcam
from services.evaluation_service import get_performance_dashboard_data
from services.ai_care_assistant import get_ai_care_response, get_curated_care_data

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('.', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Store last analysis result in memory for seamless cross-tab view
last_analysis = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main Dashboard & quick analysis endpoint."""
    if request.method == 'POST':
        return handle_upload()

    perf_data = get_performance_dashboard_data()
    return render_template(
        'dashboard.html',
        active_page='dashboard',
        last_analysis=last_analysis if last_analysis.get('result') else None,
        perf_data=perf_data
    )

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    """Dedicated MRI Analysis page."""
    if request.method == 'POST':
        return handle_upload()

    return render_template(
        'analysis.html',
        active_page='analysis',
        result=last_analysis.get('result'),
        confidence=last_analysis.get('confidence'),
        file_path=last_analysis.get('file_path'),
        prediction=last_analysis.get('prediction'),
        gradcam=last_analysis.get('gradcam'),
        probabilities=last_analysis.get('probabilities')
    )

def handle_upload(custom_file_location=None, original_filename=None):
    """Internal helper to process upload, run prediction, and compute Grad-CAM."""
    global last_analysis

    if custom_file_location:
        file_location = custom_file_location
        filename = original_filename or os.path.basename(custom_file_location)
    else:
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('analysis.html', active_page='analysis', error="Please select a valid MRI image file.")

        filename = file.filename
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_location)

    try:
        # 1. Run model prediction
        pred_data = predict_mri(file_location)

        # 2. Generate Grad-CAM pure heatmap and overlay
        gradcam_data = generate_gradcam(file_location, target_class_idx=pred_data.get('class_index'))
        gradcam_data['original_url'] = f"/uploads/{filename}"

        # 3. Cache latest analysis
        last_analysis = {
            'result': pred_data['result'],
            'confidence': pred_data['confidence'],
            'file_path': f"/uploads/{filename}",
            'prediction': pred_data,
            'gradcam': gradcam_data,
            'probabilities': pred_data['probabilities'],
            'filename': filename
        }

        return render_template(
            'analysis.html',
            active_page='analysis',
            result=pred_data['result'],
            confidence=pred_data['confidence'],
            file_path=f"/uploads/{filename}",
            prediction=pred_data,
            gradcam=gradcam_data,
            probabilities=pred_data['probabilities']
        )

    except FileNotFoundError as fnf_err:
        return render_template('analysis.html', active_page='analysis', error=str(fnf_err))
    except Exception as e:
        return render_template('analysis.html', active_page='analysis', error=f"Analysis pipeline error: {str(e)}")

@app.route('/care-assistant')
def care_assistant():
    """AI Brain Tumor Care Assistant & Educational Health Console."""
    context_category = request.args.get('context') or last_analysis.get('prediction', {}).get('top_key')
    context_display = None
    if context_category:
        context_display = CLASS_DISPLAY.get(context_category, context_category.capitalize())

    care_data = get_curated_care_data()
    return render_template(
        'care_assistant.html',
        active_page='care_assistant',
        context_category=context_category,
        context_display=context_display,
        last_analysis=last_analysis if last_analysis.get('result') else None,
        care_data=care_data
    )

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Interactive AI Chatbot endpoint answering patient & caregiver queries."""
    data = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    context_category = data.get('context') or last_analysis.get('prediction', {}).get('top_key')

    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    response = get_ai_care_response(message, context_category=context_category)
    return jsonify(response)

@app.route('/explainable-ai')
def explainable_ai():
    """Dedicated Explainable AI (Grad-CAM) deep-dive page."""
    active_gradcam = last_analysis.get('gradcam')
    active_pred = last_analysis.get('prediction')
    active_file = last_analysis.get('file_path')

    if not active_gradcam:
        sample_path = SAMPLE_IMAGES.get('glioma')
        if sample_path and os.path.exists(sample_path):
            filename = 'sample_glioma_xai.jpg'
            dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            shutil.copyfile(sample_path, dest)
            active_pred = predict_mri(dest)
            active_gradcam = generate_gradcam(dest, target_class_idx=active_pred.get('class_index'))
            active_gradcam['original_url'] = f"/uploads/{filename}"
            active_file = f"/uploads/{filename}"

    return render_template(
        'explainable_ai.html',
        active_page='explainable_ai',
        gradcam=active_gradcam,
        prediction=active_pred,
        file_path=active_file
    )

@app.route('/performance')
def performance():
    """Model Performance & Evaluation metrics dashboard."""
    perf_data = get_performance_dashboard_data()
    return render_template(
        'model_performance.html',
        active_page='performance',
        perf=perf_data
    )

@app.route('/about')
def about():
    """About project architecture, methodology, and limitations."""
    return render_template('about.html', active_page='about')

@app.route('/sample/<tumor_type>')
def test_sample(tumor_type):
    """1-Click test using validated cohort sample scans."""
    if tumor_type in SAMPLE_IMAGES:
        src = SAMPLE_IMAGES[tumor_type]
        if os.path.exists(src):
            filename = f"sample_{tumor_type}.jpg"
            dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            shutil.copyfile(src, dest)
            return handle_upload(custom_file_location=dest, original_filename=filename)
    return redirect(url_for('analysis'))

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    """Serve uploaded images securely."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/metrics')
def api_metrics():
    """REST API endpoint returning evaluation benchmark data."""
    return jsonify(get_performance_dashboard_data())

if __name__ == '__main__':
    print("[INFO] Starting NeuroScan AI Medical Platform at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
