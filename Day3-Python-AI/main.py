import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
print("===== DATASET =====")
print(df)
print("\n===== DATASET INFO =====")
print(df.info())
print("\n===== SUMMARY =====")
print(df.describe())

df.fillna(df.mean(numeric_only=True), inplace=True)
df = df[(df["Attendance"] >= 0) & (df["Attendance"] <= 100)]
print("\nAverage Assignment Score:", round(df["Assignment Score"].mean(),2))
print("Average Midterm Score:", round(df["Midterm Score"].mean(),2))
print("Average Final Score:", round(df["Final Score"].mean(),2))
highest = df.loc[df["Final Score"].idxmax()]
lowest = df.loc[df["Final Score"].idxmin()]
print("\nHighest Final Score")
print(highest)
print("\nLowest Final Score")
print(lowest)
print("\nStudents with Attendance below 75%")
print(df[df["Attendance"] < 75][["Student Name","Attendance"]])
print("\nStudents at Risk")
risk = df[(df["Final Score"] < 50) | (df["Attendance"] < 75)]
print(risk[["Student Name","Attendance","Final Score"]])
course_avg = df.groupby("Course")["Final Score"].mean()
print("\nAverage Final Score by Course")
print(course_avg)
correlation = df["Attendance"].corr(df["Final Score"])
print("\nCorrelation between Attendance and Final Score:", round(correlation,2))

# 1 histogram
plt.figure(figsize=(6,4))
plt.hist(df["Final Score"], bins=8)
plt.title("Final Score Distribution")
plt.xlabel("Final Score")
plt.ylabel("Students")
plt.savefig("score_distribution.png")
plt.close()

# 2 Bar Chart
plt.figure(figsize=(8,5))
course_avg.plot(kind="bar")
plt.title("Average Final Score by Course")
plt.xlabel("Course")
plt.ylabel("Average Score")
plt.tight_layout()
plt.savefig("average_score_by_course.png")
plt.close()

# 3 Scatter Plot
plt.figure(figsize=(6,4))
plt.scatter(df["Attendance"], df["Final Score"])
plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")
plt.savefig("attendance_vs_final_score.png")
plt.close()

print("\nCharts saved successfully!")