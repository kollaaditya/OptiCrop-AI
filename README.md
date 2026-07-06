# 🌿 OptiCrop AI – Smart Agricultural Production Optimization Engine

> AI-powered Crop Recommendation System | SmartBridge Internship Project

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)](https://flask.palletsprojects.com)
[![ML](https://img.shields.io/badge/ML-Random%20Forest-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-99%25+-brightgreen)](/)

---

## 📌 Project Description

OptiCrop AI is a full-stack Machine Learning web application that recommends the most suitable crop based on soil and climate parameters. It uses a **Random Forest classifier** trained on 2200 records across 22 crop types, achieving **99%+ accuracy**.

---

## 🎯 Features

- 🤖 **AI Crop Prediction** – Random Forest with 99%+ accuracy
- 📊 **Interactive Dashboard** – Charts, history, analytics
- 🔐 **Authentication** – Register, Login, Session management
- 👑 **Admin Panel** – User management, system analytics
- 📄 **PDF Reports** – Downloadable crop recommendation reports
- 🌙 **Dark Mode** – Full dark/light theme toggle
- 📱 **Responsive** – Mobile-first Bootstrap 5 design
- 🔍 **Search History** – Filter prediction history
- 📈 **Model Comparison** – LR, KNN, DT, RF, NB evaluated

---

## 🛠️ Tech Stack

| Layer      | Technology                          |
|------------|-------------------------------------|
| Frontend   | HTML5, CSS3, Bootstrap 5, Chart.js  |
| Backend    | Python 3.11, Flask 3.0              |
| ML         | Scikit-learn, Pandas, NumPy, Joblib |
| Database   | SQLite + Flask-SQLAlchemy           |
| Auth       | Flask-Bcrypt, Flask Sessions        |
| PDF        | ReportLab                           |
| Deploy     | Render (Gunicorn)                   |

---

## 📁 Project Structure

```
OptiCrop-AI/
├── app.py                    # Flask application
├── train_model.py            # ML training script
├── requirements.txt
├── README.md
├── dataset/
│   └── Crop_recommendation.csv
├── model/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── about.html
│   ├── predict.html
│   ├── result.html
│   ├── dashboard.html
│   ├── login.html
│   ├── register.html
│   ├── history.html
│   ├── admin.html
│   ├── contact.html
│   └── 404.html
├── static/
│   ├── css/style.css
│   ├── css/dashboard.css
│   ├── js/script.js
│   └── images/
├── database/
│   └── opticrop.db
└── notebooks/
    └── CropAnalysis.ipynb
```

---

## 🚀 Installation & Setup

### 1. Clone / Download the project
```bash
cd OptiCrop-AI
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset
Place `Crop_recommendation.csv` in the `dataset/` folder.

### 5. Train the ML model
```bash
python train_model.py
```

### 6. Run the application
```bash
python app.py
```

Open: **http://localhost:5000**

---

## 🔑 Default Credentials

| Role  | Email                | Password  |
|-------|----------------------|-----------|
| Admin | admin@opticrop.ai    | admin123  |

---

## 📊 Dataset

- **Source**: Kaggle – Crop Recommendation Dataset
- **Records**: 2200 rows × 8 columns
- **Features**: N, P, K, temperature, humidity, ph, rainfall
- **Target**: 22 crop classes (rice, maize, cotton, etc.)
- **Missing Values**: None

---

## 🤖 ML Models Compared

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~96%     |
| KNN                 | ~97%     |
| Decision Tree       | ~98%     |
| **Random Forest**   | **~99%** |
| Naive Bayes         | ~99%     |

---

## 🌐 Deployment (Render)

1. Push to GitHub
2. Create new Web Service on [render.com](https://render.com)
3. Set build command: `pip install -r requirements.txt && python train_model.py`
4. Set start command: `gunicorn app:app`

---

## 🔮 Future Enhancements

- IoT sensor integration for real-time soil data
- Mobile app (React Native)
- Live weather API integration
- Multilingual support
- GIS-based location recommendations
- Yield prediction module

---

## 👨‍💻 Author

Built for **SmartBridge Internship Program**  
Stack: Python · Flask · Scikit-learn · Bootstrap 5 · SQLite
