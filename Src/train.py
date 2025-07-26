# src/train.py

from preprocessing import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load and clean data
df, encoders = load_and_clean_data('data/StudentsPerformance.csv')

# Split features and target
X = df.drop(columns=['average_score'])
y = df['average_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Create model directory if it doesn't exist
model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
os.makedirs(model_dir, exist_ok=True)

# Save the model and encoders
model_path = os.path.join(model_dir, 'student_score_model.pkl')
encoder_path = os.path.join(model_dir, 'encoders.pkl')

joblib.dump(model, model_path)
joblib.dump(encoders, encoder_path)

print(f"✅ Model saved at: {model_path}")
print(f"✅ Encoders saved at: {encoder_path}")
