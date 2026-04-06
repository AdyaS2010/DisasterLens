# 🔍 DisasterLens AI

> **AI-Powered Disaster Response Prioritization using NLP & Machine Learning**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📋 Overview

**DisasterLens AI** is an intelligent disaster response tool that analyzes social media posts during natural disasters to **automatically classify urgency levels** and help emergency responders prioritize their efforts.

When disasters strike, thousands of social media posts flood in — cries for help, damage reports, resource requests, and general commentary. DisasterLens uses **Natural Language Processing (NLP)** and **Machine Learning** to cut through the noise, identify the most critical messages, and visualize them on an interactive map so first responders know exactly where to go first.

### Why It Matters

- 🕐 **60% of disaster deaths** occur in the first 24 hours (Red Cross)
- 📱 **72% of disaster victims** use social media to seek help (Pew Research)
- 🌍 **300M+ people** affected by natural disasters annually (UNDRR)
- 🤖 **AI can help** — automated classification reduces human bottleneck in processing thousands of messages

### Who Would Use This

- 🚒 **FEMA / Emergency managers** — prioritize rescue dispatch
- ❤️ **Red Cross volunteers** — identify where supplies are needed
- 🚓 **Local fire / police** — spot trapped or injured people faster
- 📊 **Researchers** — study disaster communication patterns

---

## 🚀 Features

### 📊 Interactive Dashboard
- Upload your own disaster CSV data or use the included sample dataset
- Interactive Plotly charts showing urgency distribution and disaster type breakdown
- Summary statistics cards (total posts, critical count, etc.)

### 🗺️ Live Disaster Map
- Interactive Folium map with color-coded urgency markers
- **🔴 Red** = Critical | **🟠 Orange** = High | **🟡 Yellow** = Medium | **🟢 Green** = Low
- **Live mode**: real-time earthquake data from USGS + natural events from NASA EONET
- **Sample mode**: geocoded locations from the training dataset
- Filter by urgency level via sidebar

### 🔍 Single Post Analyzer
- Paste any text and get instant ML-powered urgency classification
- Confidence scores with per-class probability breakdown
- Extracted locations via spaCy Named Entity Recognition
- VADER sentiment analysis with visual breakdown
- Disaster keyword detection

### 📈 Exploratory Data Analysis
- Word clouds for each urgency category
- Most common bigrams per disaster type
- Sentiment distribution charts

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Interactive web dashboard framework |
| **scikit-learn** | Machine learning (TF-IDF + Logistic Regression pipeline) |
| **spaCy** | Named Entity Recognition for location extraction |
| **NLTK** | Text preprocessing, tokenization, VADER sentiment analysis |
| **Folium** | Interactive map visualization with geocoded markers |
| **Plotly** | Dynamic charts and visualizations |
| **Pandas / NumPy** | Data manipulation and analysis |
| **Geopy** | Geocoding location names to lat/lng coordinates |
| **WordCloud** | Visual word frequency analysis |
| **Joblib** | Model serialization and loading |

---

## 📁 Project Structure

```
DisasterLens/
├── app.py                          # 🚀 Main Streamlit entry point (home page)
├── train_model.py                  # 🤖 Standalone model training script
├── requirements.txt                # 📦 Python dependencies
├── README.md                       # 📖 This file
├── .gitignore                      # 🙈 Git ignore rules
├── .streamlit/
│   └── config.toml                 # 🎨 Dark theme with emergency-red accents
├── data/
│   ├── sample_disaster_tweets.csv  # 📊 200 labeled disaster tweets
│   └── README.md                   # 📝 Dataset documentation & download links
├── models/
│   └── urgency_classifier.joblib   # 💾 Pre-trained ML model (ready to use)
├── utils/
│   ├── __init__.py
│   ├── nlp_pipeline.py             # 🔤 Text preprocessing, NER, geocoding, sentiment
│   ├── classifier.py               # 🧠 TF-IDF + Logistic Regression classifier
│   └── data_loader.py              # 📂 Data loading with Streamlit caching
└── pages/
    ├── __init__.py
    ├── 1_📊_Dashboard.py           # Dashboard with charts and file upload
    ├── 2_🗺️_Live_Map.py            # Interactive map (live + sample data)
    ├── 3_🔍_Analyze_Post.py        # Single post urgency analyzer
    └── 4_📈_EDA.py                 # Exploratory data analysis
```

---

## ⚡ Quick Start

### Prerequisites
- **Python 3.10** or higher
- **pip** (Python package manager)
- **Git**

### Option A: Run Locally (Step by Step)

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/DisasterLens.git
cd DisasterLens
```

**2. Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the spaCy English language model**
```bash
python -m spacy download en_core_web_sm
```

**5. Launch the app** 🚀
```bash
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**

> **Note:** The pre-trained model (`models/urgency_classifier.joblib`) is included in the repo. No training step needed — the app works immediately.

### Option B: Retrain the Model (Optional)

If you want to retrain the classifier (e.g., after adding more data):

```bash
python train_model.py
```

This will:
1. Load `data/sample_disaster_tweets.csv`
2. Preprocess all text through the NLP pipeline
3. Train a TF-IDF + Logistic Regression model (80/20 stratified split)
4. Print a classification report (precision, recall, F1 per class)
5. Save the new model to `models/urgency_classifier.joblib`

---

## 📊 Dataset

### Sample Data (Included)
The repo includes **200 labeled disaster tweets** (`data/sample_disaster_tweets.csv`):

| Column | Description | Example |
|---|---|---|
| `id` | Unique identifier | 1 |
| `text` | Raw tweet content | "HELP! Trapped on 2nd floor, water rising fast..." |
| `urgency` | Classification label | Critical / High / Medium / Low |
| `disaster_type` | Type of disaster | Flood / Earthquake / Hurricane / Wildfire |
| `location_mentioned` | Place name in tweet | "Houston" |

**Distribution:**
- **Critical**: 40 tweets | **High**: 50 | **Medium**: 60 | **Low**: 50
- Balanced across 4 disaster types (Flood: 51, Wildfire: 50, Hurricane: 50, Earthquake: 49)

### Using Real-World Data (For Better Accuracy)

For training on real data, download the [CrisisNLP dataset](https://crisisnlp.qcri.org/) from Kaggle (50,000+ labeled disaster tweets):

1. Download from Kaggle and place the CSV in the `data/` directory
2. Update the file path in `train_model.py`
3. Run `python train_model.py` to retrain
4. The app will automatically use the new model on next launch

### Live Data (No Download Needed)

The **Live Map** page also pulls real-time data from two free APIs:
- **USGS Earthquake API** — M2.5+ earthquakes from the last 7 days
- **NASA EONET API** — Active wildfires, volcanoes, storms, and floods worldwide

No API keys or authentication required.

---

## 🤖 How the ML Pipeline Works

```
Raw Tweet Text
      │
      ▼
┌───────────────────┐
│ Text Preprocessing │  Remove URLs, @mentions, #symbols
│   (NLTK + regex)   │  Lowercase, tokenize, remove stopwords
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   TF-IDF          │  Convert to 5,000-dim feature vector
│   Vectorizer      │  Unigrams + bigrams (captures "flash flood")
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Logistic        │  Multi-class classifier (4 urgency levels)
│   Regression      │  Outputs probability scores per class
└────────┬──────────┘
         │
         ▼
   Urgency Label + Confidence Score
```

### Additional NLP Analysis (runs in parallel):

| Technique | Tool | Purpose |
|---|---|---|
| **Sentiment Analysis** | NLTK VADER | Compound score (-1 to +1) detects distress level |
| **Named Entity Recognition** | spaCy (en_core_web_sm) | Extracts GPE/LOC entities (cities, states) |
| **Geocoding** | Geopy Nominatim | Converts location names → lat/lng for map |

### Model Performance (on 200 sample tweets)

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| **Critical** | 0.80 | 0.50 | 0.62 |
| **High** | 0.60 | 0.30 | 0.40 |
| **Medium** | 0.48 | 1.00 | 0.65 |
| **Low** | 1.00 | 0.50 | 0.67 |
| **Overall Accuracy** | | | **60%** |

> With the full CrisisNLP dataset (50K+ tweets), accuracy would improve significantly.

---

## 🧪 Try It Out

Paste these into the **Analyze Post** page:

| Text | Expected Result |
|---|---|
| *"HELP! Trapped on roof, water rising fast in Galveston! Children here!"* | 🔴 Critical |
| *"Need insulin refrigerated ASAP. Without power 3 days at shelter."* | 🟠 High |
| *"Major structural damage on 5th Ave after the quake."* | 🟡 Medium |
| *"Praying for everyone in Florida tonight. Stay safe."* | 🟢 Low |

---

## 🌐 Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo and select `app.py` as the entry point
4. Add to **Advanced settings** → **Packages**:
   ```
   en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz
   ```
5. Deploy — your app will be live at `https://your-app.streamlit.app`

---

## 🙏 Acknowledgments

- **[CrisisNLP](https://crisisnlp.qcri.org/)** — Crisis-related tweet datasets and research
- **[USGS Earthquake Hazards](https://earthquake.usgs.gov/)** — Real-time earthquake data API
- **[NASA EONET](https://eonet.gsfc.nasa.gov/)** — Earth Observatory Natural Event Tracker
- **[Streamlit](https://streamlit.io/)** — For making data apps easy to build
- **[scikit-learn](https://scikit-learn.org/)** — Machine learning made accessible
- **[spaCy](https://spacy.io/)** — Industrial-strength NLP
- **[NLTK](https://www.nltk.org/)** — The classic NLP toolkit

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Adya Sastry

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  🔍 <b>DisasterLens AI</b> — <em>When every second counts, AI prioritizes lives.</em>
</p>
