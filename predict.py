import argparse

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Score transactions for fraud risk.")
    parser.add_argument("--input", required=True, help="CSV of transactions to score")
    parser.add_argument("--output", default="scored.csv", help="Where to write scored output")
    parser.add_argument(
        "--model", default="models/random_forest.joblib", help="Path to trained model"
    )
    parser.add_argument("--scaler", default="models/scaler.joblib", help="Path to fitted scaler")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability cutoff for flagging a transaction as fraud",
    )
    args = parser.parse_args()

    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)

    df = pd.read_csv(args.input)
    X = df.drop(columns=["Class"], errors="ignore").copy()
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

    probs = model.predict_proba(X)[:, 1]
    df["fraud_probability"] = probs
    df["flagged_as_fraud"] = (probs >= args.threshold).astype(int)

    df.to_csv(args.output, index=False)
    n_flagged = df["flagged_as_fraud"].sum()
    print(f"Scored {len(df):,} transactions -> {n_flagged} flagged as fraud "
          f"(threshold={args.threshold}). Written to {args.output}")


if __name__ == "__main__":
    main()
