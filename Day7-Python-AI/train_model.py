import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("dataset.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
df = df[(df["Attendance"] >= 0) & (df["Attendance"] <= 100)]
df["Pass"] = (df["Final Score"] >= 50).astype(int)
X = df[
    [
        "Attendance",
        "Assignment Score",
        "Midterm Score",
        "Final Score"
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
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
print("Model saved successfully!")