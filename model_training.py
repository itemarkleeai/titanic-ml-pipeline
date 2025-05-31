import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load cleaned data
df = pd.read_csv("Git_titanic_cleaned.csv")

features = df.drop(["Survived", "PassengerId"], axis=1)
target = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "rf_model.pkl")
