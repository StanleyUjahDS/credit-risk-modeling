"""Leakage-aware, interpretable credit-risk modelling pipeline."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data():
    dataset = fetch_openml("credit-g", version=1, as_frame=True)
    x = dataset.data
    y = (dataset.target == "good").astype(int)
    return x, y


def build_pipeline(x):
    numeric = x.select_dtypes(include="number").columns
    categorical = x.select_dtypes(exclude="number").columns
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def main():
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    pipeline = build_pipeline(x)
    pipeline.fit(x_train, y_train)
    probability = pipeline.predict_proba(x_test)[:, 1]
    prediction = pipeline.predict(x_test)
    print({"roc_auc": round(float(roc_auc_score(y_test, probability)), 4)})
    print(classification_report(y_test, prediction))


if __name__ == "__main__":
    main()
