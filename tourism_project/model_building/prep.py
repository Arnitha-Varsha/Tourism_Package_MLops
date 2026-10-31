import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_PATH = "/content/tourism_project/data/tourism.csv"

# Load dataset
df = pd.read_csv(RAW_PATH, on_bad_lines="skip")

# Clean column inconsistencies as requested
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Drop unnecessary columns (e.g., CustomerID)
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# Split into features (X) and target (y)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the splits to CSV files locally
X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("Data prepared, cleaned, and split successfully!")
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
