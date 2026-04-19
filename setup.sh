# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh — Environment Bootstrap Script
#
# Downloads required NLP models and corpora (spaCy English model, NLTK
# tokenizers and lexicons). Run once after installing Python dependencies.
# ──────────────────────────────────────────────────────────────────────────────

python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
