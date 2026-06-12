# NeuroScan - AI-Powered Brain Tumor Detection

## Overview

NeuroScan is a deep learning-based web application designed to assist in the detection and classification of brain tumors from MRI scans. The system uses a Convolutional Neural Network (CNN) trained on brain MRI images to classify scans into multiple categories and provides visual explanations using Grad-CAM heatmaps.

The project combines Machine Learning, Medical Image Processing, Flask API development, and a responsive web interface to create an end-to-end diagnostic assistance platform.

---

## Features

- Brain MRI image classification
- Multi-class tumor detection
- Grad-CAM visualization for model interpretability
- Drag-and-drop image upload
- Multiple image analysis support
- Confidence score prediction
- Probability distribution for all classes
- Session history using local storage
- Responsive frontend interface
- Flask REST API backend

---

## Tumor Classes

The model classifies MRI scans into:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## Project Structure

```text
NeuroScan/
│
├── backend/
│   ├── app.py
│   ├── model/
│   │   └── class_indices.json
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── notebooks/
│   ├── train_model.py
│   ├── training_original.py
│   └── plots/
│       ├── confusion_matrix.png
│       └── training_history.png
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

### Machine Learning

- TensorFlow
- Keras
- NumPy
- Scikit-learn

### Backend

- Flask
- Flask-CORS
- Gunicorn

### Frontend

- HTML5
- CSS3
- JavaScript

### Visualization

- Matplotlib
- Seaborn
- Grad-CAM

### Image Processing

- OpenCV
- Pillow

---

## Installation

### Clone Repository

```bash
git clone https://github.com/SamikshaLakshman/NeuroScan.git
cd NeuroScan
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Backend

Navigate to the backend folder:

```bash
cd backend
```

Run Flask server:

```bash
python app.py
```

The API will start on:

```text
http://localhost:5000
```

---

## Running the Frontend

Open:

```text
frontend/index.html
```

in your browser.

Ensure the Flask backend is running before making predictions.

---

## API Endpoint

### Predict Brain Tumor

**POST**

```http
/predict
```

#### Request

Multipart form-data:

```text
images: MRI image files
```

#### Response

```json
{
  "filename": "scan.jpg",
  "predicted": "Glioma",
  "confidence": 0.97,
  "probabilities": {
    "Glioma": 0.97,
    "Meningioma": 0.01,
    "Pituitary Tumor": 0.01,
    "No Tumor": 0.01
  }
}
```

---

## Model Explainability

NeuroScan integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight regions of MRI scans that contribute most to the model's prediction.

This improves transparency and interpretability of the AI system.

---

## Future Enhancements

- Cloud deployment
- User authentication
- PDF diagnostic reports
- Patient record management
- Advanced MRI visualization
- Mobile-friendly application
- Real-time database integration

---

## Disclaimer

This project is developed for educational and research purposes only.

It is not intended to replace professional medical diagnosis, treatment, or clinical decision-making.

---

## Authors

### Team Members

- Prajwal H S
- Samiksha Lakshman Swamy

---

## License

This project is released for academic and educational use.
