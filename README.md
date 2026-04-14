# 🌟 AI Stress Prediction System (FastAPI + ML)

## 📌 Overview

This project is an end-to-end AI system that predicts a user's stress level based on daily lifestyle data.

The system uses a Machine Learning model (KNN) trained on features such as sleep, work, physical activity, and steps.
It also includes a FastAPI backend and a user-friendly frontend interface.

---

## 🚀 Features

* 🔍 Predict stress level (Low / Medium / High)
* ⚡ Real-time prediction via FastAPI API
* 📊 Feature influence visualization
* 🎯 Clean and interactive UI
* 🧠 Machine Learning pipeline with scaling and tuning

---

## 🧠 Machine Learning Model

* Algorithm: **K-Nearest Neighbors (KNN)**
* Preprocessing: **StandardScaler**
* Hyperparameter tuning: **GridSearchCV**
* Evaluation: Accuracy, Confusion Matrix, Cross Validation

---

## 📂 Project Structure

```
project/
│
├── main.py              
├── model.pkl           
├── scaler.pkl          
├── stress_data_fake.csv 
├── index.html          
└── README.md           
```

---

## ⚙️ Installation

### Install dependencies

```
pip install pandas numpy scikit-learn fastapi uvicorn joblib
```

---

## ▶️ Running the Project

### Start the server

```
uvicorn main:app --reload
```

Server will run at:
http://127.0.0.1:8000

### Open the UI

Open `index.html` in your browser.

---

## 🔄 How It Works

1. User enters data in the UI
2. JavaScript sends POST request to FastAPI
3. Backend:

   * Converts input to array
   * Applies scaling
   * Runs ML model
4. Server returns prediction
5. UI displays result + feature influence

---

## 📊 Example Input

```
{
  "Sleep_Hours": 7,
  "Work_Hours": 8,
  "Steps": 6000,
  "Exercise_Minutes": 30
}
```

---

## 📈 Future Improvements

* Improve model accuracy with more data
* Try advanced models (Random Forest / XGBoost)
* Deploy to cloud
* Mobile-friendly UI

---

## 👩‍💻 Author

Developed as part of a Machine Learning project combining AI, backend, and frontend development.

---

## ⭐ Notes

This project demonstrates understanding of:

* Machine Learning workflow
* API development with FastAPI
* Full-stack integration
