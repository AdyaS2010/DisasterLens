# CrisisNLP Training Data

## About CrisisNLP

[CrisisNLP](https://crisisnlp.qcri.org/) is a research initiative providing
labeled crisis-related social media data for training NLP models.

## Obtaining the Data

### Option 1: HuggingFace Datasets (Recommended)

The CrisisNLP datasets are available on HuggingFace Hub. The training script
can work with any CSV that has `text` and `urgency`/`label` columns.

Recommended datasets:
- `crisis_nlp` — Multi-event crisis data
- `crisisbench` — Benchmark dataset for crisis informatics

### Option 2: Direct Download

1. Visit [crisisnlp.qcri.org](https://crisisnlp.qcri.org/)
2. Download the "CrisisNLP Volunteers" or "CrisisLexT26" dataset
3. Place the CSV file(s) in this `data/` directory

### Option 3: Use the Built-in Sample Data

The included `sample_disaster_tweets.csv` (200 rows) can be used for
quick testing of the training pipeline:

```bash
python train_bert_model.py --data data/sample_disaster_tweets.csv --epochs 3
```

## Training the DistilBERT Model

```bash
# Using sample data (quick test)
python train_bert_model.py

# Using CrisisNLP data
python train_bert_model.py --data data/crisisnlp_data.csv --epochs 5 --batch-size 16

# Full options
python train_bert_model.py --help
```

The trained model will be saved to `models/bert/` and automatically
loaded by the DisasterLens app on next startup.

## Label Mapping

The training script automatically maps CrisisNLP labels to our 4 urgency levels:

| CrisisNLP Label | DisasterLens Urgency |
|---|---|
| `injured_or_dead_people` | Critical |
| `missing_or_found_people` | Critical |
| `requests_or_urgent_needs` | Critical |
| `infrastructure_and_utility_damage` | High |
| `affected_individuals` | High |
| `displaced_people_and_evacuations` | High |
| `rescue_volunteering_or_donation_effort` | Medium |
| `caution_and_advice` | Medium |
| `other_relevant_information` | Low |
| `sympathy_and_support` | Low |
| `not_humanitarian` | Low |
