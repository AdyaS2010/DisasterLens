# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# train_model.py — TF-IDF Urgency Classifier Training Script
#
# Standalone CLI script that trains a TF-IDF + Logistic Regression pipeline
# on the sample disaster tweets CSV. Produces a serialized .joblib model file
# in models/ that the app loads at runtime as the lightweight fallback
# classifier when DistilBERT is unavailable.
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.classifier import train_model, MODEL_PATH

SAMPLE_DATA_PATH = os.path.join("data", "sample_disaster_tweets.csv")


def main():
    print("=" * 60)
    print("  DisasterLens — Urgency Classifier Training")
    print("=" * 60)
    print()

    print(f"Loading data from: {SAMPLE_DATA_PATH}")

    if not os.path.exists(SAMPLE_DATA_PATH):
        print(
            f"\nERROR: Sample data file not found at '{SAMPLE_DATA_PATH}'.\n"
            f"Make sure you have the sample dataset in the data/ folder."
        )
        sys.exit(1)

    df = pd.read_csv(SAMPLE_DATA_PATH)
    print(f"Loaded {len(df)} rows.\n")

    required_columns = ["text", "urgency"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    print("Starting model training...\n")
    model = train_model(df)

    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Model saved to: {os.path.abspath(MODEL_PATH)}")
    print()
    print("  You can now run the DisasterLens Streamlit app and it")
    print("  will automatically load this trained model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
