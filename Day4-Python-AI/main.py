
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

df = pd.read_csv("dataset.csv")
print("===== FIRST 5 RECORDS =====")
print(df.head())
df.fillna(df.mean(numeric_only=True), inplace=True)
df = df[(df["Attendance"] >= 0) & (df["Attendance"] <= 100)]
df["Pass"] = df["Final Score"].apply(lambda x: 1 if x >= 50 else 0)
print("\nDataset after adding target column:\n")
print(df.head())
# Create target
df["Pass"] = (df["Final Score"] >= 50).astype(int)

# Features
X = df[
    [
        "Attendance",
        "Assignment Score",
        "Midterm Score"
    ]
]

y = df["Pass"]

print(df["Pass"].value_counts())
y = df["Pass"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\nPredictions:")
print(predictions)
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix")
cm = confusion_matrix(y_test, predictions)
print(cm)
print("\nClassification Report")
print(classification_report(y_test, predictions))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()
prediction_counts = pd.Series(predictions).value_counts()
plt.figure(figsize=(5,4))
prediction_counts.plot(kind="bar")
plt.title("Prediction Distribution")
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.xticks([0,1],["Fail","Pass"], rotation=0)
plt.tight_layout()
plt.savefig("prediction_distribution.png")
plt.close()
print("\nCharts saved successfully!")