import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
df = pd.read_csv("dataset.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
df = df[(df["Attendance"] >= 0) & (df["Attendance"] <= 100)]
df["Pass"] = (df["Final Score"] >= 50).astype(int)
print("\nClass Distribution")
print(df["Pass"].value_counts())
X = df[
    [
        "Attendance",
        "Assignment Score",
        "Midterm Score"
    ]
]
y = df["Pass"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=4
)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
def evaluate(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n===== {model_name} =====")
    print("Accuracy :", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall   :", round(recall, 3))
    print("F1 Score :", round(f1, 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(
        f"{model_name.lower().replace(' ','_')}_confusion_matrix.png"
    )
    plt.close()
    return [accuracy, precision, recall, f1]

log_results = evaluate(
    "Logistic Regression",
    y_test,
    log_pred
)
tree_results = evaluate(
    "Decision Tree",
    y_test,
    tree_pred
)
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        log_results[0],
        tree_results[0]
    ],
    "Precision": [
        log_results[1],
        tree_results[1]
    ],
    "Recall": [
        log_results[2],
        tree_results[2]
    ],
    "F1 Score": [
        log_results[3],
        tree_results[3]
    ]
})
print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(comparison)
plt.figure(figsize=(8,5))
plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()
print("\nCharts saved successfully!")