import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
df = pd.read_csv("dataset.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
df = df[(df["Attendance"] >= 0) &
        (df["Attendance"] <= 100)]
df["Average Score"] = (
    df["Assignment Score"] +
    df["Midterm Score"] +
    df["Final Score"]
) / 3
df["Pass"] = (df["Final Score"] >= 50).astype(int)
X = df[
    [
        "Attendance",
        "Assignment Score",
        "Midterm Score",
        "Average Score"
    ]
]
y = df["Pass"]
print("\nClass Distribution")
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier(random_state=42))
])
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_prob = baseline.predict_proba(X_test)[:,1]
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier(random_state=42))
])
params = {
    "model__criterion": ["gini","entropy"],
    "model__max_depth": [2,3,4,5,None],
    "model__min_samples_split":[2,4,6],
    "model__min_samples_leaf":[1,2,3]
}
grid = GridSearchCV(
    pipeline,
    param_grid=params,
    cv=5,
    scoring="accuracy"
)
grid.fit(X_train,y_train)
best_model = grid.best_estimator_
print("\nBest Parameters")
print(grid.best_params_)
tuned_pred = best_model.predict(X_test)
tuned_prob = best_model.predict_proba(X_test)[:,1]
def evaluate(name,true,pred,prob):
    acc = accuracy_score(true,pred)
    pre = precision_score(true,pred,zero_division=0)
    rec = recall_score(true,pred,zero_division=0)
    f1 = f1_score(true,pred,zero_division=0)
    roc = roc_auc_score(true,prob)
    print("\n",name)
    print("Accuracy :",round(acc,3))
    print("Precision:",round(pre,3))
    print("Recall   :",round(rec,3))
    print("F1 Score :",round(f1,3))
    print("ROC AUC  :",round(roc,3))
    return [acc,pre,rec,f1,roc]
baseline_scores = evaluate(
    "Baseline Model",
    y_test,
    baseline_pred,
    baseline_prob
)
tuned_scores = evaluate(
    "Tuned Model",
    y_test,
    tuned_pred,
    tuned_prob
)
disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test,baseline_pred)
)
disp.plot()
plt.title("Before Tuning")
plt.savefig("confusion_matrix_before.png")
plt.close()
disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test,tuned_pred)
)
disp.plot()
plt.title("After Tuning")
plt.savefig("confusion_matrix_after.png")
plt.close()

RocCurveDisplay.from_predictions(
    y_test,
    tuned_prob
)
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.close()
comparison = pd.DataFrame({
"Metric":[
"Accuracy",
"Precision",
"Recall",
"F1 Score",
"ROC AUC"
],
"Before Tuning":baseline_scores,
"After Tuning":tuned_scores
})
print("\nComparison Table")
print(comparison)
comparison.set_index("Metric").plot(
kind="bar",
figsize=(8,5)
)
plt.ylabel("Score")
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig("comparison_chart.png")
plt.close()
print("\nCharts saved successfully!")