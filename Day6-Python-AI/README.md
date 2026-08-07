# Day 6 – Model Optimization using Feature Engineering & Hyperparameter Tuning

## Objective

Improve the student performance prediction model using feature engineering, feature scaling, and hyperparameter tuning.

---

## Workflow

1. Load dataset
2. Clean data
3. Create new feature (Average Score)
4. Select important features
5. Scale features using StandardScaler
6. Split into training and testing sets
7. Train baseline Decision Tree model
8. Optimize model using GridSearchCV
9. Compare both models

---

## Features Used

- Attendance
- Assignment Score
- Midterm Score
- Average Score

Target:

- Pass = 1
- Fail = 0

---

## Hyperparameter Tuning

GridSearchCV was used to search for the best:

- Criterion
- Max Depth
- Minimum Samples Split
- Minimum Samples Leaf

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Visualizations

- Confusion Matrix Before Tuning
- Confusion Matrix After Tuning
- ROC Curve
- Model Comparison Chart

---

## What I Learned

- Importance of feature engineering
- Feature scaling using StandardScaler
- Hyperparameter tuning with GridSearchCV
- How tuning can improve model performance
- Comparing baseline and optimized models

---

## Run

```bash
pip install -r requirements.txt

python main.py
```