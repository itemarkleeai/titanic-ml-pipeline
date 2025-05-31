import pandas as pd

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Fill missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# Feature engineering: FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Encode 'Sex'
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Drop unused columns
df = df.drop(["Cabin", "Ticket", "Name", "Embarked"], axis=1)

# Save cleaned data (optional)
df.to_csv("Git_titanic_cleaned.csv", index=False)
