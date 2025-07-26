# src/predict.py

import joblib
import numpy as np

# Sample input (replace these values with actual data as needed)
# Format: [gender, race/ethnicity, parental level of education, lunch, test preparation course, math score, reading score, writing score]

# You must encode the categorical values exactly like during training
# Example input (already encoded): [1, 2, 3, 1, 0, 78, 72, 70]
sample_input = np.array([[1, 2, 3, 1, 0, 78, 72, 70]])  # Shape: (1, 8)

# Load the trained model
model = joblib.load('model/student_score_model.pkl')

# Predict
predicted_score = model.predict(sample_input)
print("🎯 Predicted average score:", predicted_score[0])
