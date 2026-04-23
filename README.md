# 🍫 Cocoa Pod Disease Detection

A deep learning-powered web application that detects cocoa pod diseases from images using a custom CNN model built with PyTorch and served via a Flask web interface.

---

## 📌 Overview

Cocoa farming is a vital agricultural industry, but pod diseases can cause significant crop losses. This application helps farmers and agricultural workers quickly identify diseases by simply uploading a photo of a cocoa pod — getting an instant prediction and actionable recommendations.

---

## 🎯 Disease Classes

The model can detect the following conditions:

| Class | Description |
|---|---|
| 🟤 Black Pod Rot | Fungal disease caused by *Phytophthora* species |
| ❄️ Frosty Pod Rot | Caused by the fungus *Moniliophthora roreri* |
| 🐛 Mirid | Damage caused by mirid bugs (capsid insects) |
| ✅ Healthy | No disease detected |

---

## 🧠 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 89% |

> Model trained on dataset sourced from [Roboflow](https://roboflow.com)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | PyTorch (Custom CNN) |
| Backend | Python, Flask |
| Frontend | HTML, CSS |
| Database | MySQL |
| AI Assistant | Groq API |

---

## ✨ Features

- 📸 Upload a cocoa pod image and get an instant disease prediction
- 📋 View detailed disease information and treatment recommendations
- 👤 User registration and login system
- 📊 Dashboard to track prediction history
- 🔐 Secure admin panel

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- MySQL Server
- A Groq API Key (free at [https://console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/m-safrinn/cocoa-pod-disease-detection.git
cd cocoa-pod-disease-detection
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up your API key**

Open `app.py` and replace:
```python
GROQ_API_KEY = "ADD YOUR API KEY HERE"
```
with your actual Groq API key.

5. **Set up the MySQL database**

Create a MySQL database and update the database credentials in `app.py`.

6. **Run the application**
```bash
python app.py
```

7. Open your browser and go to `http://localhost:5000`

---

## 📁 Project Structure

```
cocoa-pod-disease-detection/
│
├── templates/                  # HTML templates
│   ├── index.html              # Landing page
│   ├── detect.html             # Disease detection page
│   ├── dashboard.html          # User dashboard
│   └── admin.html              # Admin panel
│
├── app.py                      # Main Flask application
├── cocoa_model_bundle.pt       # Trained PyTorch CNN model
├── test_app.py                 # Unit tests
└── requirements.txt            # Python dependencies
```

---

## 👨‍💻 Author

**Mohamad Safrin**
BSc Data Science

---

## 📄 License

This project is intended for academic and educational purposes.
