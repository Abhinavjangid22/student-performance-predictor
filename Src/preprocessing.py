# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Create target column
    df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)

    # Encode categorical columns
    label_cols = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders
