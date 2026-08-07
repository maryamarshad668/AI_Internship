from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

app = FastAPI(title="Student Performance Prediction API")
model = joblib.load("model.pkl")
class Student(BaseModel):
    Attendance: float = Field(..., ge=0, le=100)
    Assignment_Score: float = Field(..., ge=0, le=100)
    Midterm_Score: float = Field(..., ge=0, le=100)
    Final_Score: float = Field(..., ge=0, le=100)
@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API"
    }
@app.post("/predict")
def predict(student: Student):
    data = [[
        student.Attendance,
        student.Assignment_Score,
        student.Midterm_Score,
        student.Final_Score
    ]]
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]
    confidence = round(max(probability) * 100, 2)
    return {
        "prediction": "Pass" if prediction == 1 else "Fail",
        "confidence": f"{confidence}%"
    }