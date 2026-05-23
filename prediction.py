import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load("logistic_regression_pipeline.joblib")

print("Enter student details for prediction:\n")

# Take user input13

self_reported_confidence_python = float(
    input("Self-reported confidence in Python (numeric): ")
)
practice_problems_solved = int(
    input("Number of practice problems solved: ")
)
hours_spent_learning_per_week = float(
    input("Hours spent learning per week: ")
)
projects_completed = int(
    input("Number of projects completed: ")
)

# Create DataFrame (must match training feature names & order)
input_data = pd.DataFrame(
    [[
        self_reported_confidence_python,
        practice_problems_solved,
        hours_spent_learning_per_week,
        projects_completed
    ]],
    columns=[
        "self_reported_confidence_python",
        "practice_problems_solved",
        "hours_spent_learning_per_week",
        "projects_completed"
    ]
)

# Make prediction
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

# Output result
if prediction == 1:
    print("\nPrediction: Student is likely to PASS the exam")
else:
    print("\nPrediction: Student is likely to FAIL the exam")

print(f"Confidence (Fail, Pass): {prediction_proba}")
