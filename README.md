# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions
using the classic Kaggle credit card fraud dataset (284,807 transactions,
492 confirmed frauds — a highly imbalanced 0.17% positive rate).

## Project structure

```
fraud_detection_project/
├── data/
│   └── creditcard.csv          # dataset (place it here)
├── models/                     # trained models saved here after running
├── outputs/                    # plots + metrics saved here after running
├── train_model.py              # trains + evaluates the models
├── predict.py                  # scores new transactions with the saved model
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place `creditcard.csv` in the `data/` folder.

## Train the models

```bash
python train_model.py --data data/creditcard.csv
```

This will:
1. Split the data 70/30 (stratified, so the fraud rate is preserved in both sets).
2. Standardize the `Time` and `Amount` columns (the `V1`-`V28` columns are
   already PCA-transformed and don't need scaling).
3. Train two models with class-weighting to handle the imbalance:
   - **Logistic Regression** — a fast, interpretable baseline.
   - **Random Forest** — the stronger model, better precision/recall tradeoff.
4. Evaluate with ROC-AUC, PR-AUC, confusion matrix, and per-class precision/recall/F1
   (accuracy alone is misleading on a dataset this imbalanced).
5. Save trained models to `models/`, and metrics + plots to `outputs/`:
   - `results.json` — all metrics for both models
   - `feature_importance.csv` — Random Forest feature ranking
   - `curves.png` — precision-recall and ROC curves
   - `feature_importance.png` — top 12 most predictive features

## Score new transactions

```bash
python predict.py --input data/new_transactions.csv --output outputs/scored.csv --threshold 0.5
```

The input CSV needs the same columns as the training data (`Time`, `V1`-`V28`,
`Amount`), without `Class`. Output adds `fraud_probability` and `flagged_as_fraud`
columns. Lower the `--threshold` to catch more fraud at the cost of more false
positives, or raise it to reduce false alarms.

## Results (reference run)

| Metric              | Logistic Regression | Random Forest |
|----------------------|---------------------|----------------|
| ROC-AUC              | 0.968               | 0.970          |
| PR-AUC               | 0.700               | **0.792**      |
| Precision (fraud)    | 6.7%                | **86.1%**      |
| Recall (fraud)       | 87.8%               | 75.0%          |
| F1 (fraud)           | 0.125               | **0.801**      |
| False positives      | 1,806               | **18**         |

Random Forest is the better production choice: logistic regression catches
slightly more fraud but at the cost of an unusable number of false alarms.

The five most predictive features are `V14`, `V4`, `V10`, `V17`, and `V12`
(anonymized PCA components — the dataset doesn't expose real-world feature
names like merchant category or location).

## Notes / next steps

- The default classification threshold is 0.5 — tune it in `predict.py` based
  on whether false positives (blocked legitimate purchases) or false negatives
  (missed fraud) are more costly for your use case.
- Because `V1`-`V28` are PCA-anonymized, this project is best used to
  benchmark model architecture and imbalance-handling techniques rather than
  to derive interpretable business rules.
