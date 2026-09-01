import os
import pandas as pd


SOURCE = "data/creditcard.csv"
OUTPUT = "data/demo_transactions.csv"


def main():

    if not os.path.exists(SOURCE):
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE}"
        )

    df = pd.read_csv(SOURCE)

    required_columns = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
        "Class"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # Take a few examples of each class.
    legit = (
        df[df["Class"] == 0]
        .sample(n=5, random_state=42)
    )

    fraud = (
        df[df["Class"] == 1]
        .sample(n=5, random_state=42)
    )

    demo = pd.concat(
        [legit, fraud],
        ignore_index=True
    )

    demo = demo.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    os.makedirs(
        "data",
        exist_ok=True
    )

    demo.to_csv(
        OUTPUT,
        index=False
    )

    print(
        f"Created {OUTPUT} "
        f"with {len(demo)} transactions."
    )


if __name__ == "__main__":
    main()