import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned data and model
df = pd.read_csv("Git_titanic_cleaned.csv")
model = joblib.load("rf_model.pkl")

features = df.drop(["Survived", "PassengerId"], axis=1)
target = df["Survived"]

# (For demo, use all data for evaluation)
predictions = model.predict(features)

print("Accuracy:", accuracy_score(target, predictions))
print(classification_report(target, predictions))
