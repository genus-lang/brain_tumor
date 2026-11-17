# NeuroScan AI | Clinical Brain MRI Diagnostic & Interpretability Platform

NeuroScan AI is an end-to-end deep learning, Explainable AI (XAI), and clinical health education web application. It combines **VGG16 Transfer Learning** for 4-class brain tumor detection, **Grad-CAM** visual saliency maps, empirical performance benchmarking, and an **AI Brain Tumor Care Assistant** designed to help patients, caregivers, and families understand brain tumors, treatments, safety precautions, emergency red flags, and recovery.

---

## 🌟 Key Platform Capabilities

1. **VGG16 Transfer Learning Backbone:**
   - 16-layer convolutional neural network fine-tuned on clinical brain MRI cohorts.
   - Input dimension: $128 \times 128 \times 3$ RGB normalized tensor.
   - Staged fine-tuning with frozen foundational feature blocks and adaptable `block5` convolutional filters.

2. **Explainable AI (Grad-CAM):**
   - **Target Layer:** Final convolutional layer (`block5_conv3`).
   - Dynamic gradient calculation of the predicted class score with respect to convolutional feature maps.
   - Global average pooled channel weights ($\alpha_k^c$) with ReLU linear combination.
   - High-resolution bilinear resampling and standard clinical JET colormap overlay.
   - **Interactive 3-Way Multi-Modal Viewer:** Smoothly toggle between **Original MRI**, **Pure Heatmap**, and **Grad-CAM Overlay**.

3. **Interactive Prediction Analytics (Chart.js):**
   - 4-class calibrated differential probability bar chart based on the model's actual softmax output vector.
   - Circular progress confidence meter.
   - Complete scan metadata extraction (resolution, file size, dimensions, format).

4. **Model Performance & Empirical Benchmarks:**
   - Evaluated on holdout test cohort with **90.0% Test Accuracy**, **89.99% Precision**, **90.0% Recall**, and **89.83% F1-score**.
   - Interactive Training vs. Validation Accuracy curve (8 epochs).
   - Interactive Training vs. Validation Loss curve.
   - Empirical $4 \times 4$ Confusion Matrix heatmap.
   - Full scikit-learn Classification Report table.

5. **AI Brain Tumor Care Assistant (`/care-assistant`):**
   - **Interactive Health Chatbot:** Answers natural language questions on treatments, symptoms, diet, safety, and surgery.
   - **Curated Knowledge Base:** Sourced from **NCI**, **WHO**, **American Cancer Society (ACS)**, and **NHS UK**.
   - **Post-Prediction Integration:** One-click `"Explore Patient Care & Guidance"` button automatically provides educational context based on the classified scan.
   - **Treatment Explorer:** Deep-dives into Surgery (craniotomy/biopsy), Radiation (SRS/IMRT), Chemotherapy (Temozolomide), Targeted Therapy, Active Surveillance, and Supportive Care.
   - **Safety Precautions:** Medication compliance, anti-seizure protocols, fall prevention, balanced nutrition (explicit warning: no diet or remedy cures tumors).
   - **Emergency Triage:** Immediate emergency signs (first seizure, status epilepticus, paralysis) vs. prompt evaluation signs.
   - **Questions for Doctors:** Printable checklist for clinical consultations.
   - **Dual-Mode AI:** Fully functional offline out-of-the-box; supports optional `GEMINI_API_KEY` for live generative responses.

---

## 🧠 Supported Diagnostic Classes

| Pathology Category | Anatomical Origin | Typical MRI Presentation |
| :--- | :--- | :--- |
| **Glioma** | Glial support cells | Intra-axial infiltrative lesion with indistinct margins and mass effect |
| **Meningioma** | Arachnoid meninges | Extra-axial, well-circumscribed dural mass causing cortical compression |
| **Pituitary Adenoma** | Sella turcica / pituitary | Skull-base lesion with superior expansion toward the optic chiasm |
| **No Tumor (Healthy)** | Physiological brain | Symmetric ventricles, intact midline alignment, no edema or lesions |

---

## 📁 System Architecture

```text
brain_tumor/
├── app.py                             # Clean Flask server launcher
├── main.py                            # Flask application routes & REST endpoints
├── train.py                           # Model training and fine-tuning pipeline
├── predict.py                         # CLI inference script
├── models/
│   └── model.h5                       # Trained VGG16 model weights
├── services/
│   ├── __init__.py
│   ├── prediction_service.py          # Model caching, preprocessing & inference
│   ├── gradcam_service.py             # Grad-CAM backprop, JET mapping & overlay
│   ├── evaluation_service.py          # Metrics, confusion matrix & history loader
│   └── ai_care_assistant.py           # Clinical knowledge base, NLP matcher & Gemini API
├── templates/
│   ├── base.html                      # Global healthcare SaaS layout & navigation
│   ├── index.html                     # Forwarding template
│   ├── dashboard.html                 # Main dashboard with KPIs and fast upload
│   ├── analysis.html                  # Scan analysis, Chart.js, Grad-CAM & Care CTA
│   ├── care_assistant.html            # AI Care Assistant: Chatbot, Treatments, Precautions
│   ├── explainable_ai.html            # Grad-CAM mathematics & deep-dive viewer
│   ├── model_performance.html         # Empirical graphs, confusion matrix & report
│   └── about.html                     # VGG16 architecture breakdown & limitations
├── static/
│   ├── css/
│   │   ├── style.css                  # Healthcare SaaS design system
│   │   └── care_assistant.css         # Care Assistant & Chatbot styling
│   ├── js/
│   │   ├── main.js                    # Upload validation & drag-and-drop
│   │   ├── heatmap.js                 # 3-way multi-modal viewer controller
│   │   ├── charts.js                  # Chart.js probability, accuracy & loss graphs
│   │   └── care_assistant.js          # Chatbot logic, tabs & question chips
│   ├── results/                       # Generated Grad-CAM heatmaps & overlays
│   └── images/                        # Static UI assets
├── reports/
│   ├── evaluation_metrics.json        # Test set benchmarks & confusion matrix
│   └── training_history.json          # Training & validation epoch curves
├── dataset/                           # Kaggle dataset (Training/ & Testing/)
├── uploads/                           # Temporary uploaded scan storage
├── requirements.txt                   # Project dependencies
└── README.md                          # Documentation
```

---

## 🚀 Running the Project Locally (VS Code / PowerShell)

### 1. Open Terminal in VS Code
Press `Ctrl + ~` to open the integrated terminal in the project root:
```powershell
cd d:\dev\AI_ML\brain_tumor
```

### 2. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. (Optional) Set Gemini API Key
The assistant works 100% offline out-of-the-box using the verified NCI/WHO/ACS knowledge engine. If you would like live Gemini generative responses, set your API key:
```powershell
$env:GEMINI_API_KEY="your-gemini-api-key-here"
```

### 4. Start the NeuroScan AI Web Server
Run either launcher:
```powershell
python app.py
# or
python main.py
```

### 5. Open the Web Application
Open your web browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🔬 Explainable AI: Grad-CAM Formulation

1. **Gradient Computation:**
   $$\frac{\partial y^c}{\partial A^k}$$
   Calculates the rate of change of score $y^c$ with respect to filter $k$ in `block5_conv3`.

2. **Global Average Pooling:**
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}$$
   Yields the importance weight $\alpha_k^c$ for each channel.

3. **ReLU Combination:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
   Filters out negative contributions and keeps regions with positive influence on the target class.

---

## ⚖️ Important Medical Disclaimer

> **Important:** This application is an academic and research-oriented brain MRI image classification project. Its predictions, Grad-CAM visualizations, and AI Care Assistant responses are educational and are not intended for clinical diagnosis, treatment decisions, or medical prescriptions. Always consult qualified healthcare professionals for medical evaluation.
