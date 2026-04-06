import re
import time

import nltk

for _resource in ["punkt", "punkt_tab", "stopwords", "vader_lexicon"]:
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

_spacy_nlp = None


def _load_spacy_model():
    """Load and cache the spaCy English model."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp

    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
        return _spacy_nlp
    except ImportError:
        print("spaCy is not installed. Run: pip install spacy")
        return None
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        return None


_geocode_cache = {}
_STOPWORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """Clean and tokenize text: remove URLs, mentions, stopwords, lowercase."""
    if not text or not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.lower()

    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()

    tokens = [
        word for word in tokens
        if word.isalpha() and word not in _STOPWORDS
    ]

    return " ".join(tokens)


def extract_locations(text: str) -> list:
    """Extract location names from text using spaCy NER (GPE/LOC entities)."""
    if not text or not isinstance(text, str):
        return []

    nlp = _load_spacy_model()
    if nlp is None:
        return []

    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]


def geocode_location(location: str) -> tuple:
    """Convert a location name to (lat, lon) using Nominatim. Results are cached."""
    if not location or not isinstance(location, str):
        return None

    if location in _geocode_cache:
        return _geocode_cache[location]

    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="disasterlens_app")
        time.sleep(1)  # respect rate limits

        result = geolocator.geocode(location)
        if result:
            coords = (result.latitude, result.longitude)
            _geocode_cache[location] = coords
            return coords
        else:
            _geocode_cache[location] = None
            return None

    except ImportError:
        print("geopy is not installed. Run: pip install geopy")
        return None
    except Exception as e:
        print(f"Geocoding failed for '{location}': {e}")
        return None


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment using VADER with disaster-context adjustments.

    VADER misreads words like "rescue" and "help" as positive, but in
    disaster contexts they signal danger. We apply a penalty when crisis
    indicators appear alongside these misleadingly-positive words.
    """
    default = {
        "compound": 0.0,
        "pos": 0.0,
        "neg": 0.0,
        "neu": 1.0,
        "label": "Neutral",
    }

    if not text or not isinstance(text, str):
        return default

    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]

        # Disaster-context adjustment: penalize misleadingly-positive words
        # when crisis indicators are present
        text_lower = text.lower()

        misleading_positive = [
            "rescue", "save", "saved", "saving", "urgent", "urgently",
            "help", "relief", "survive", "survived", "shelter",
            "evacuate", "evacuation", "emergency", "immediate",
        ]

        crisis_indicators = [
            "trapped", "flood", "hurricane", "earthquake", "wildfire",
            "fire", "tornado", "tsunami", "storm", "collapsed",
            "destroyed", "damage", "injured", "missing", "stranded",
            "drowning", "buried", "rubble", "debris", "casualties",
            "sos", "mayday", "need help", "please help", "send help",
            "water rising", "no way out", "stuck", "dying",
        ]

        has_misleading = any(w in text_lower for w in misleading_positive)
        crisis_count = sum(1 for w in crisis_indicators if w in text_lower)

        if has_misleading and crisis_count >= 1:
            penalty = -0.15 * crisis_count
            compound = max(-1.0, compound + penalty)

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "compound": round(compound, 4),
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
            "label": label,
        }

    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        return default
