# Student Performance Prediction API

## Overview

This project deploys a Machine Learning model as a REST API using FastAPI. The API predicts whether a student will pass or fail based on academic performance.

---

## Technologies

- Python
- FastAPI
- Scikit-learn
- Pandas
- Joblib

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the API:

```bash
uvicorn app:app --reload
```

---

## Endpoint

### POST /predict

Request:

```json
{
    "Attendance": 90,
    "Assignment_Score": 80,
    "Midterm_Score": 75,
    "Final_Score": 85
}
```

Response:

```json
{
    "prediction": "Pass",
    "confidence": "99.72%"
}
```

---

## Features

- Model loading
- Input validation
- JSON responses
- Prediction confidence
- Error handling using FastAPI validation

---

## Testing

The API was tested using:

- FastAPI Swagger UI (`/docs`)
- Postman