# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# utils/bert_classifier.py — DistilBERT Urgency Classifier
#
# Handles loading, inference, and batch prediction using a fine-tuned
# DistilBERT model for 4-class urgency classification. Supports loading
# from a local directory or HuggingFace Hub, with graceful degradation
# when transformers/torch are not installed.
# ──────────────────────────────────────────────────────────────────────────────

"""
DisasterLens AI — DistilBERT Urgency Classifier
Fine-tuned on CrisisNLP data for 4-class urgency classification.
Falls back gracefully if transformers/torch are not installed.
"""

import os
import logging

logger = logging.getLogger(__name__)

BERT_MODEL_DIR = os.path.join("models", "bert")
# Public HuggingFace Hub model ID — set this after you push a trained model
HF_MODEL_ID = None  # e.g. "AdyaS2010/disasterlens-urgency-bert"

LABEL_MAP = {0: "Critical", 1: "High", 2: "Medium", 3: "Low"}
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)


def _is_available():
    """Check whether transformers + torch are importable."""
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def load_bert_model(model_path=None):
    """
    Load a fine-tuned DistilBERT model and tokenizer.

    Priority:
        1. Local directory (models/bert/)
        2. HuggingFace Hub (if HF_MODEL_ID is set)
        3. Return None (let caller fall back to TF-IDF)

    Returns:
        tuple: (model, tokenizer) or (None, None)
    """
    if not _is_available():
        logger.info("transformers/torch not installed — skipping BERT loader")
        return None, None

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    path = model_path or BERT_MODEL_DIR

    # 1. Try local directory
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
        logger.info("Loading DistilBERT from local directory: %s", path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.eval()
            return model, tokenizer
        except Exception as e:
            logger.warning("Failed to load local BERT model: %s", e)

    # 2. Try HuggingFace Hub
    if HF_MODEL_ID:
        logger.info("Downloading DistilBERT from HuggingFace Hub: %s", HF_MODEL_ID)
        try:
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
            model.eval()
            # Cache locally for next time
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            logger.info("Saved BERT model to %s for future use", path)
            return model, tokenizer
        except Exception as e:
            logger.warning("Failed to download from HF Hub: %s", e)

    logger.info("No BERT model available — will fall back to TF-IDF")
    return None, None


def predict_urgency_bert(text: str, model, tokenizer) -> tuple:
    """
    Predict urgency for a single text using the DistilBERT model.

    Returns:
        tuple: (predicted_label: str, confidence: float)
    """
    if not text or model is None or tokenizer is None:
        return ("Low", 0.0)

    import torch

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        predicted_id = torch.argmax(probs).item()
        confidence = probs[predicted_id].item()
        label = LABEL_MAP.get(predicted_id, "Medium")

        return (label, confidence)

    except Exception as e:
        logger.error("BERT prediction failed: %s", e)
        return ("Medium", 0.0)


def predict_urgency_bert_batch(texts: list, model, tokenizer, batch_size=16) -> list:
    """
    Predict urgency for a batch of texts.

    Returns:
        list of tuples: [(label, confidence), ...]
    """
    if not texts or model is None or tokenizer is None:
        return [("Low", 0.0)] * len(texts)

    import torch

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            for j in range(len(batch)):
                pred_id = torch.argmax(probs[j]).item()
                conf = probs[j][pred_id].item()
                results.append((LABEL_MAP.get(pred_id, "Medium"), conf))
        except Exception as e:
            logger.error("BERT batch prediction failed: %s", e)
            results.extend([("Medium", 0.0)] * len(batch))

    return results


def get_bert_probabilities(text: str, model, tokenizer) -> dict:
    """
    Get per-class probabilities for a single text.

    Returns:
        dict: {"Critical": 0.xx, "High": 0.xx, "Medium": 0.xx, "Low": 0.xx}
    """
    if not text or model is None or tokenizer is None:
        return {label: 0.25 for label in LABEL_MAP.values()}

    import torch

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        return {
            LABEL_MAP[i]: round(probs[i].item(), 4)
            for i in range(NUM_LABELS)
        }
    except Exception:
        return {label: 0.25 for label in LABEL_MAP.values()}
