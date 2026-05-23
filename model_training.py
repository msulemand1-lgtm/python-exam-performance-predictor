import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

data = pd.read_csv("./cleaned_data.csv")

X = data[
    [
        "self_reported_confidence_python",
        "practice_problems_solved",
        "hours_spent_learning_per_week",
        "projects_completed"
    ]
]

y = data["passed_exam"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]
)


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

joblib.dump(pipeline, "logistic_regression_pipeline.joblib")

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
