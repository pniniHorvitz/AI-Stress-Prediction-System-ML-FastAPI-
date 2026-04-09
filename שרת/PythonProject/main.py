# =============================
# 1️⃣ ייבוא ספריות
# =============================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib

# =============================
# 2️⃣ טעינת המודל וה-Scaler
# =============================
# טוענים את המודל והסקיילר פעם אחת בלבד בעת העלאת השרת
knn_model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# =============================
# 3️⃣ יצירת האפליקציה
# =============================
app = FastAPI(
    title="Stress Level Prediction API",
    description="Predicts stress level based on Sleep, Work, Steps, and Exercise",
    version="1.0"
)

# =============================
# 4️⃣ הגדרת CORS
# =============================
# מאפשר דפדפן לשלוח בקשות לשרת (HTML/JS) ללא שגיאות אבטחה
origins = [
    "*",  # אפשר גם להכניס את כתובת האתר שלך במקום *
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST וכו'
    allow_headers=["*"],
)


# =============================
# 5️⃣ מחלקת BaseModel – נתוני כניסה
# =============================
# השמות חייבים להיות זהים לשדות ב-frontend
class StressInput(BaseModel):
    Sleep_Hours: float
    Work_Hours: float
    Steps: float
    Exercise_Minutes: float

#11
# =============================
# 6️⃣ Endpoint לניבוי
# =============================
@app.post("/predict")
def predict_stress(data: StressInput):
    """
    Predict stress level based on user input
    """
    # 1️⃣ הפיכת הנתונים ל-Array
    input_array = np.array([[data.Sleep_Hours,
                             data.Work_Hours,
                             data.Steps,
                             data.Exercise_Minutes]])

    # 2️⃣ Scaling
    input_scaled = scaler.transform(input_array)

    # 3️⃣ Prediction
    pred_class = knn_model.predict(input_scaled)[0]

    # 4️⃣ החזרת התוצאה כ-JSON
    # 0 = Low, 1 = Medium, 2 = High
    stress_mapping = {0: "Low", 1: "Medium", 2: "High"}

    return {"predicted_stress_level": stress_mapping[pred_class]}


# =============================
# 🌟 אתגר מתקדם - Endpoint נוסף להחזרת חשיבות פיצ'רים
# =============================
@app.get("/info")
def model_info():
    """
    Returns feature importance for the model
    KNN does not have coef_ like linear models,
    but ניתן להראות את מספר השכנים הטובים ביותר
    """
    return {
        "model_type": "KNN Classifier",
        "best_n_neighbors": knn_model.n_neighbors,
        "features": ["Sleep_Hours", "Work_Hours", "Steps", "Exercise_Minutes"]
    }


@app.post("/feature_influence")
def feature_influence(data: StressInput):
    """
    מחזיר את התחזית והפיצ'ר שהכי השפיע על התחזית,
    כולל אחוזי השפעה לכל פיצ'ר
    """
    # 1️⃣ הפיכת הנתונים ל-Array
    X_user = np.array([[data.Sleep_Hours, data.Work_Hours, data.Steps, data.Exercise_Minutes]])

    # 2️⃣ Scaling
    X_scaled = scaler.transform(X_user)
#22
    # 3️⃣ שולפים שכנים קרובים
    neighbors = knn_model.kneighbors(X_scaled, n_neighbors=knn_model.n_neighbors, return_distance=True)
    indices = neighbors[1]  # אינדקסים של השכנים
    X_train_scaled = knn_model._fit_X  # הנתונים המקוריים של ה-train set

    # 4️⃣ חישוב contribution של כל פיצ'ר
    contributions = np.mean(np.abs(X_scaled - X_train_scaled[indices]), axis=1)
    mean_contrib = np.mean(contributions, axis=0)

    feature_names = ["Sleep_Hours", "Work_Hours", "Steps", "Exercise_Minutes"]

    # 5️⃣ המרת contributions לאחוזים
    total = np.sum(mean_contrib)
    influence_percent = {feature_names[i]: round((mean_contrib[i] / total) * 100, 2) for i in range(len(feature_names))}

    most_influential = max(influence_percent, key=influence_percent.get)

    # 6️⃣ חיזוי התחזית והמרה ל-Low/Medium/High
    pred_class = knn_model.predict(X_scaled)[0]
    stress_mapping = {0: "Low", 1: "Medium", 2: "High"}

    return {
        "predicted_stress_level": stress_mapping[pred_class],
        "most_influential_feature": most_influential,
        "feature_influence_percent": influence_percent
    }

#uvicorn main:app --reload :להרצה