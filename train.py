import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load Dataset

df = pd.read_csv("dataset/health_dataset.csv")

# Encode Gender

df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

# Encode BMI Category

df["BMI Category"] = df["BMI Category"].map({
    "Normal": 0,
    "Overweight": 1,
    "Obese": 2
})

# Create Target Column

def health_risk(row):

    if row["Stress Level"] <= 4 and row["Sleep Duration"] >= 7:
        return 0

    elif row["Stress Level"] <= 7:
        return 1

    else:
        return 2

df["Risk"] = df.apply(health_risk, axis=1)

# Features

X = df[[
    "Gender",
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Heart Rate",
    "Daily Steps"
]]

# Target

y = df["Risk"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model

model = RandomForestClassifier()

model.fit(X_train, y_train)

# Save Model

pickle.dump(model, open("model.pkl", "wb"))

print("Model Trained Successfully")
