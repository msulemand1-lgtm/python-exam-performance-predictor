import pandas as pd
import numpy as np

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

data = pd.read_csv("./python_learning_exam_performance.csv")

data2 = data.dropna().copy()

## We are dropping Student_id and country name column because as we know student id does not matter and we can't say the student had passed the exam because he/she is from a specific country
data2.drop(columns=["student_id", "country"], inplace=True)

df_encoded = data2.copy()

categorical_cols = ["prior_programming_experience"]

le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    
    
final_data = df_encoded[[
'self_reported_confidence_python',
'practice_problems_solved',
'hours_spent_learning_per_week',
'projects_completed', 'passed_exam']].copy()

final_data.to_csv('cleaned_data.csv')