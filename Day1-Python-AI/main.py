# Student Information Management System

students = []

def add_student():
    name = input("Enter student name: ")
    age = int(input("Enter age: "))
    marks = float(input("Enter marks: "))
    if marks >= 50:
        result = "Pass"
    else:
        result = "Fail"
    student = {
        "Name": name,
        "Age": age,
        "Marks": marks,
        "Result": result
    }
    students.append(student)
    print("Student added successfully\n")
def display_students():
    if len(students) == 0:
        print("No student records found.\n")
        return
    print("\n----- Student Records -----")
    for student in students:
        print(f"Name   : {student['Name']}")
        print(f"Age    : {student['Age']}")
        print(f"Marks  : {student['Marks']}")
        print(f"Result : {student['Result']}")
        print("--------------------------")

while True:
    print("\nStudent Management System")
    print("1. Add Student")
    print("2. View Students")
    print("3. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        add_student()
    elif choice == "2":
        display_students()
    elif choice == "3":
        print("\nFinal Student Records:")
        display_students()
        print("Program Ended.")
        break
    else:
        print("Invalid choice! Try again.")