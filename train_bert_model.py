# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# train_bert_model.py — DistilBERT Fine-Tuning Script
#
# Fine-tunes distilbert-base-uncased for 4-class urgency classification
# (Critical / High / Medium / Low). Supports CrisisNLP CSV data or the
# built-in sample dataset. Outputs a HuggingFace-compatible model directory
# to models/bert/ that the app auto-detects at startup.
# ──────────────────────────────────────────────────────────────────────────────

"""
DisasterLens AI — DistilBERT Training Script

Fine-tunes distilbert-base-uncased for 4-class urgency classification.
Can use either:
    1. Local CrisisNLP CSV data
    2. The built-in sample_disaster_tweets.csv (200 rows, for quick testing)

Usage:
    python train_bert_model.py                          # Use sample data
    python train_bert_model.py --data path/to/data.csv  # Use custom CSV
    python train_bert_model.py --epochs 5 --batch-size 8

The CSV must have at least a 'text' column and an 'urgency' column
(or labels that can be mapped to Critical/High/Medium/Low).
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BERT_MODEL_DIR = os.path.join("models", "bert")
SAMPLE_DATA_PATH = os.path.join("data", "sample_disaster_tweets.csv")


def map_urgency_label(label: str) -> str:
    """Normalize various urgency/severity labels to our 4 standard classes."""
    if not isinstance(label, str):
        return "Medium"

    label_lower = label.strip().lower()
    mapping = {
        "critical": "Critical", "crit": "Critical", "severe": "Critical",
        "emergency": "Critical", "urgent": "Critical", "very high": "Critical",
        "high": "High", "important": "High", "significant": "High",
        "moderate": "Medium", "medium": "Medium", "med": "Medium",
        "normal": "Medium", "standard": "Medium",
        "low": "Low", "minor": "Low", "info": "Low",
        "informational": "Low", "minimal": "Low", "none": "Low",
        # CrisisNLP specific mappings
        "not_humanitarian": "Low",
        "other_relevant_information": "Low",
        "infrastructure_and_utility_damage": "High",
        "affected_individuals": "High",
        "injured_or_dead_people": "Critical",
        "missing_or_found_people": "Critical",
        "rescue_volunteering_or_donation_effort": "Medium",
        "caution_and_advice": "Medium",
        "sympathy_and_support": "Low",
        "requests_or_urgent_needs": "Critical",
        "displaced_people_and_evacuations": "High",
    }

    if label_lower in mapping:
        return mapping[label_lower]

    for key, val in mapping.items():
        if key in label_lower:
            return val

    return "Medium"


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for disaster urgency classification"
    )
    parser.add_argument(
        "--data", type=str, default=SAMPLE_DATA_PATH,
        help="Path to CSV file with 'text' and 'urgency' columns"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--max-length", type=int, default=256,
        help="Max token length (default: 256)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=BERT_MODEL_DIR,
        help=f"Output directory for trained model (default: {BERT_MODEL_DIR})"
    )
    args = parser.parse_args()

    # ── Check dependencies ─────────────────────────────────────────────────
    try:
        import torch
        import pandas as pd
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from datasets import Dataset
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        import numpy as np
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with: pip install transformers torch datasets scikit-learn pandas numpy")
        sys.exit(1)

    print("=" * 60)
    print("  DisasterLens — DistilBERT Urgency Classifier Training")
    print("=" * 60)
    print()

    # ── Load data ──────────────────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows.\n")

    # ── Find columns ───────────────────────────────────────────────────────
    text_col = None
    for candidate in ["text", "tweet_text", "tweet", "message", "content"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            avg_lens = {c: df[c].astype(str).str.len().mean() for c in str_cols}
            text_col = max(avg_lens, key=avg_lens.get)

    urgency_col = None
    for candidate in ["urgency", "urgency_label", "label", "priority", "severity",
                       "class", "classification", "choose_one_category"]:
        if candidate in df.columns:
            urgency_col = candidate
            break

    if text_col is None or urgency_col is None:
        print(f"ERROR: Could not find text/urgency columns.")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Text column: '{text_col}'")
    print(f"Label column: '{urgency_col}'")

    # ── Prepare labels ─────────────────────────────────────────────────────
    df = df[[text_col, urgency_col]].dropna().copy()
    df.columns = ["text", "urgency_raw"]
    df["urgency"] = df["urgency_raw"].apply(map_urgency_label)

    # Label encoding
    label_map = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    df["label"] = df["urgency"].map(label_map)

    print(f"\nMapped urgency distribution:")
    print(df["urgency"].value_counts().to_string())
    print()

    # ── Split ──────────────────────────────────────────────────────────────
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples:  {len(test_df)}\n")

    # ── Tokenize ───────────────────────────────────────────────────────────
    model_name = "distilbert-base-uncased"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading model: {model_name} (num_labels={len(label_map)})")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_map)
    )

    # ── Training ───────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...\n")
    trainer.train()

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Classification Report:")
    print("=" * 50)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = test_df["label"].values

    id_to_label = {v: k for k, v in label_map.items()}
    pred_names = [id_to_label[p] for p in preds]
    true_names = [id_to_label[t] for t in true_labels]

    print(classification_report(true_names, pred_names))

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nModel saved to: {os.path.abspath(args.output_dir)}")
    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Model directory: {os.path.abspath(args.output_dir)}")
    print()
    print("  The DisasterLens app will automatically detect and use")
    print("  this model on next startup.")
    print("=" * 60)


if __name__ == "__main__":
    main()
