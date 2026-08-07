# Day 5 – Model Comparison

## Objective

This project compares two Machine Learning classification models for predicting whether a student will pass or fail.

---

## Models Used

- Logistic Regression
- Decision Tree Classifier

---

## Features Used

- Attendance
- Assignment Score
- Midterm Score

---

## Target

- Pass = 1
- Fail = 0

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Visualizations

- Logistic Regression Confusion Matrix
- Decision Tree Confusion Matrix
- Model Accuracy Comparison

---

## Files

- dataset.csv
- main.py
- requirements.txt
- README.md

---

## How to Run

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python main.py
```

---

## Analysis

### Which model performed better?

Compare the Accuracy, Precision, Recall, and F1 Score after running the program.

### Why?

- Logistic Regression works well for linear relationships.
- Decision Trees can capture more complex patterns but may overfit small datasets.

### Was the dataset balanced?

Check:

```python
print(df["Pass"].value_counts())
```

A balanced dataset improves model reliability.

### What could improve the model?

- More training data
- Better feature selection
- Hyperparameter tuning
- Cross-validation
- Additional relevant student performance features

---

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib