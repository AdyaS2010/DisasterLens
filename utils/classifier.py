import os

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.nlp_pipeline import preprocess_text

MODEL_PATH = os.path.join("models", "urgency_classifier.joblib")


def build_pipeline() -> Pipeline:
    """Create a TF-IDF + Logistic Regression pipeline."""
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
            ),
        ),
    ])
    return pipeline


def train_model(df: pd.DataFrame) -> Pipeline:
    """Train the urgency classifier on a labeled DataFrame with 'text' and 'urgency' columns."""
    print("Preprocessing text...")
    X = df["text"].apply(preprocess_text)
    y = df["urgency"]

    print(f"Dataset size: {len(df)} tweets")
    print(f"Urgency distribution:\n{y.value_counts().to_string()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}\n")

    print("Training model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Classification Report:")
    print("-" * 50)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    return pipeline


def load_model():
    """Load a trained model from disk, or return None if not found."""
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    else:
        print(f"No saved model found at: {MODEL_PATH}")
        return None


def predict_urgency(text: str, model) -> tuple:
    """Predict urgency level for a single text. Returns (label, confidence)."""
    if not text or model is None:
        return ("Low", 0.0)

    cleaned = preprocess_text(text)
    if not cleaned.strip():
        return ("Low", 0.0)

    predicted_label = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    confidence = float(max(probabilities))

    return (predicted_label, confidence)


def load_or_train_model(df: pd.DataFrame):
    """Load a saved model if available, otherwise train a new one."""
    model = load_model()
    if model is not None:
        return model

    print("No saved model found. Training a new one...")
    return train_model(df)
