"""
app.py - NeuroScan AI Entry Point
Runs the Flask server backed by VGG16 Deep Learning model.
"""

from main import app

if __name__ == '__main__':
    print("[INFO] Starting NeuroScan AI Web Server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

