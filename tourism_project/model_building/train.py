import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

# 1. Load data splits
X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")
y_train = pd.read_csv("ytrain.csv").values.ravel()
y_test = pd.read_csv("ytest.csv").values.ravel()

# Identify numerical and categorical features
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

# 2. Setup preprocessing pipelines
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# 3. Create full pipeline with XGBoost classifier
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"))
])

# 4. Define simple parameter set for training
mlflow.set_experiment("Tourism_Package_Prediction")

with mlflow.start_run():
    # Fit pipeline
    clf.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_param("random_state", 42)

    # Predict
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)

    print(f"Model evaluation complete. Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}")

    # Log model artifacts to MLflow
    mlflow.sklearn.log_model(clf, "model")

    # Save local artifact to deployment directory
    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(clf, "tourism_project/deployment/model.pkl")
    print("Model saved locally to tourism_project/deployment/model.pkl")
