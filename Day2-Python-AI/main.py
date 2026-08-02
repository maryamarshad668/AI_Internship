import pandas as pd

students = [
    {"Name": "Maryam", "Age": 20, "Course": "Computer Science", "Marks": 85},
    {"Name": "Sara", "Age": 21, "Course": "Software Engineering", "Marks": 72},
    {"Name": "Ahmed", "Age": 19, "Course": "Information Technology", "Marks": 91},
    {"Name": "Fatima", "Age": 22, "Course": "Computer Science", "Marks": 68},
    {"Name": "Hassan", "Age": 20, "Course": "Artificial Intelligence", "Marks": 77},
    {"Name": "Ayesha", "Age": 21, "Course": "Data Science", "Marks": 95},
    {"Name": "Bilal", "Age": 23, "Course": "Cyber Security", "Marks": 59},
    {"Name": "Zain", "Age": 20, "Course": "Computer Science", "Marks": 81},
    {"Name": "Noor", "Age": 22, "Course": "Software Engineering", "Marks": 74},
    {"Name": "Usman", "Age": 19, "Course": "Artificial Intelligence", "Marks": 65}
]

df = pd.DataFrame(students)

print("********* ALL STUDENTS *********")
print(df)
print("********* STUDENTS WITH MARKS ABOVE 70 *********")
print(df[df["Marks"] > 70])
average_marks = df["Marks"].mean()
print("Average Marks:", round(average_marks, 2))
highest = df.loc[df["Marks"].idxmax()]
print("********* HIGHEST MARKS *********")
print(highest)
lowest = df.loc[df["Marks"].idxmin()]
print("********* LOWEST MARKS *********")
print(lowest)
print("Total Students:", len(df))