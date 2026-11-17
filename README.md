# Brain Tumor Detection Using Deep Learning (VGG16 Transfer Learning)

An end-to-end Deep Learning and Computer Vision application to detect and classify Brain Tumors from MRI images into 4 distinct classes with high accuracy (~95%+), accompanied by an interactive Flask web application.

---

## 🧠 Overview & Tumor Classes

The model classifies brain MRI scans into:
1. **Glioma (`glioma`)**
2. **Meningioma (`meningioma`)**
3. **Pituitary Tumor (`pituitary`)**
4. **Healthy / No Tumor (`notumor`)**

### Architecture:
- **Base Model:** Pre-trained **VGG16** (ImageNet weights, input shape `128x128x3`)
- **Fine-Tuning:** Freezes earlier layers and unfreezes the last convolution blocks
- **Classification Head:** `Flatten` -> `Dropout(0.3)` -> `Dense(128, ReLU)` -> `Dropout(0.2)` -> `Dense(4, Softmax)`
- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam(learning_rate=0.0001)`

---

## 📁 Project Structure

```text
brain_tumor/
├── brain_tumour_detection_using_deep_learning.ipynb  # Training & evaluation notebook
├── main.py                                           # Flask web backend & prediction API
├── templates/
│   └── index.html                                    # Web UI upload interface
├── models/
│   └── model.h5                                      # Saved trained model weights (place here)
├── uploads/                                          # Uploaded MRI scans
├── requirements.txt                                  # Python dependencies
├── README.md                                         # Project documentation & guide
├── Te-gl_0015.jpg                                    # Sample test image (Glioma)
├── Te-meTr_0001.jpg                                  # Sample test image (Meningioma)
├── Te-noTr_0004.jpg                                  # Sample test image (No Tumor)
└── Te-piTr_0003.jpg                                  # Sample test image (Pituitary)
```

---

## 🚀 Quick Setup Guide

### Step 1: Create and Activate Virtual Environment

Open a terminal or PowerShell in this project directory:

```bash
# On Windows (PowerShell / Command Prompt):
python -m venv .venv
.\.venv\Scripts\activate

# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Obtain or Place the Trained Model (`models/model.h5`)

The web app requires the trained model file at `models/model.h5`.

#### Option A: Train via Google Colab (Recommended - Free GPU)
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `brain_tumour_detection_using_deep_learning.ipynb`.
3. In Colab, enable GPU runtime: **Runtime** -> **Change runtime type** -> **T4 GPU**.
4. Download the [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and extract `Training/` and `Testing/` to your Google Drive or Colab runtime directory.
5. Run the notebook cells to train the model.
6. The notebook saves `model.h5`. Download this file to your computer and place it in the `models/` directory:
   ```text
   brain_tumor/models/model.h5
   ```

#### Option B: If you already have `model.h5`
Simply copy your `model.h5` file into the `models/` folder.

---

### Step 4: Run the Web Application

With your virtual environment activated:

```bash
python main.py
```

The application will start at:
👉 **`http://127.0.0.1:5000`**

---

### Step 5: Test the Application

1. Open your browser and navigate to `http://127.0.0.1:5000`.
2. Click **"Choose File"** and select one of the provided sample test images:
   - `Te-gl_0015.jpg` (Expected: **Tumor: glioma**)
   - `Te-meTr_0001.jpg` (Expected: **Tumor: meningioma**)
   - `Te-noTr_0004.jpg` (Expected: **No Tumor**)
   - `Te-piTr_0003.jpg` (Expected: **Tumor: pituitary**)
3. Click **"Upload and Detect"**.
4. The system will display the diagnosis result along with the confidence percentage!
