# DisasterLens AI — Data Directory

## Sample Dataset: `sample_disaster_tweets.csv`

A synthetic dataset of **200 realistic disaster-related social media posts** for development, testing, and model prototyping. These are **not real tweets** — they are generated to mimic the tone, structure, and content of actual crisis communications on social media.

### Schema

| Column | Type | Description |
|---|---|---|
| `id` | integer | Unique row identifier (1–200) |
| `text` | string | The tweet/post text with realistic social media language, hashtags, and abbreviations |
| `urgency` | string | Urgency classification: `Critical`, `High`, `Medium`, or `Low` |
| `disaster_type` | string | One of: `Hurricane`, `Earthquake`, `Flood`, `Wildfire` |
| `location_mentioned` | string | A real US city or region mentioned in the post |

### Urgency Distribution

| Urgency | Count | Description |
|---|---|---|
| **Critical** (~20%) | 40 | Life-threatening: trapped people, medical emergencies, imminent danger |
| **High** (~25%) | 50 | Immediate needs: shelter, water, rescue requests, medical supplies |
| **Medium** (~30%) | 60 | Damage reports, supply logistics, status updates, volunteer coordination |
| **Low** (~25%) | 50 | Prayers, news sharing, commentary, preparedness tips, donations |

### Disaster Type Distribution

Each disaster type has ~50 posts (25%): Hurricane, Earthquake, Flood, Wildfire.

### Location Coverage

Posts reference real US cities and regions including Houston, Miami, San Francisco, Los Angeles, Nashville, Portland, Anchorage, Fort Myers, New Orleans, Seattle, and many more.

---

## Real-World Dataset: CrisisNLP

For production training and evaluation, we recommend the **CrisisNLP** dataset, a widely-used benchmark for crisis informatics and disaster NLP research.

### How to Download

1. **From CrisisNLP website:**
   - Visit [https://crisisnlp.qcri.org/](https://crisisnlp.qcri.org/)
   - Navigate to the "Datasets" section
   - Download the relevant crisis tweet collections

2. **From Kaggle (curated subsets):**
   - Search for "CrisisNLP" or "crisis tweets" on [Kaggle Datasets](https://www.kaggle.com/datasets)
   - Popular options include:
     - [CrisisMMD](https://www.kaggle.com/datasets) — multimodal crisis dataset
     - [Natural Disaster Tweets](https://www.kaggle.com/datasets) — labeled disaster tweets
   - Download via the Kaggle UI or CLI:
     ```bash
     pip install kaggle
     kaggle datasets download -d <dataset-slug>
     ```

3. **From HuggingFace Hub:**
   - Search for crisis/disaster datasets on [HuggingFace Datasets](https://huggingface.co/datasets)
   - Load directly in Python:
     ```python
     from datasets import load_dataset
     dataset = load_dataset("<dataset-name>")
     ```

### CrisisNLP Dataset Details

The CrisisNLP resource includes multiple crisis-related datasets:

- **CrisisLexT26** — ~28,000 tweets from 26 crisis events, labeled by informativeness and information type
- **CrisisLexT6** — ~60,000 tweets from 6 crisis events with finer-grained labels
- **CrisisMMD** — Multimodal (text + image) crisis data from 7 disaster events
- **ASONAM17** — Domain adaptation dataset across crisis types

### Typical Schema (CrisisNLP)

| Column | Description |
|---|---|
| `tweet_id` | Original Twitter status ID |
| `tweet_text` | Raw tweet text |
| `label` | Informativeness label (e.g., "informative" / "not informative") |
| `info_type` | Information type (e.g., "affected individuals", "infrastructure damage", "donations") |
| `crisis_event` | Name of the crisis event |

---

## Citation

If you use CrisisNLP data in your research, please cite:

```bibtex
@inproceedings{imran2016lrec,
  title={LREC 2016 Tutorial: Automatic Processing of Micro-Blog Messages for Detection of Situational Awareness Information during Crises},
  author={Imran, Muhammad and Mitra, Prasenjit and Castillo, Carlos},
  booktitle={Proceedings of LREC},
  year={2016}
}

@article{alam2018crisismmd,
  title={CrisisMMD: Multimodal Twitter Datasets from Natural Disasters},
  author={Alam, Firoj and Ofli, Ferda and Imran, Muhammad},
  journal={Proceedings of the International AAAI Conference on Web and Social Media},
  year={2018}
}

@inproceedings{imran2013practical,
  title={Practical Extraction of Disaster-Relevant Information from Social Media},
  author={Imran, Muhammad and Elbassuoni, Shady and Castillo, Carlos and Diaz, Fernando and Meier, Patrick},
  booktitle={Proceedings of the 22nd International Conference on World Wide Web (WWW)},
  year={2013}
}
```

---

## Usage in DisasterLens

```python
import pandas as pd

# Load sample data for development
df = pd.read_csv("data/sample_disaster_tweets.csv")

# Quick stats
print(df["urgency"].value_counts())
print(df["disaster_type"].value_counts())

# Filter critical tweets
critical = df[df["urgency"] == "Critical"]
print(f"Critical tweets: {len(critical)}")
```
