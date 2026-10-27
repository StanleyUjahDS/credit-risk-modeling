# Credit Risk Modelling

An interpretable classification project for credit-risk modelling using the
public OpenML `credit-g` dataset. It demonstrates a reproducible preprocessing
pipeline for mixed numerical and categorical data.

## Approach

- Fetch the public dataset at runtime with scikit-learn.
- Impute missing values and one-hot encode categorical variables.
- Train a regularised logistic-regression model.
- Evaluate ROC-AUC, precision, recall, and confusion-matrix metrics.
- Keep the workflow explainable and suitable for governance review.

This is an educational model and must not be used for lending decisions. Real
deployment would require independent validation, bias testing, explainability,
monitoring, regulatory review, and human oversight.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.model
pytest
```

The raw dataset is not committed; it is downloaded from OpenML at runtime.
