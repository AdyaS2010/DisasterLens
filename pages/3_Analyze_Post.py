import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Analyze Post | DisasterLens AI",
    page_icon="🔍",
    layout="wide",
)

from utils.theme import (
    inject_css, sidebar_nav, footer, get_colors,
    URGENCY_COLORS, URGENCY_EMOJI, URGENCY_ORDER, COLORS,
)
from utils.classifier import load_or_train_model, predict_urgency, load_model
from utils.nlp_pipeline import extract_locations, analyze_sentiment, preprocess_text
from utils.data_loader import load_sample_data

inject_css()
sidebar_nav()

URGENCY_STYLES = {
    "Critical": {"bg": COLORS["critical"], "fg": "#FFFFFF", "emoji": "🔴", "glow": "rgba(255,59,59,0.3)"},
    "High":     {"bg": COLORS["high"],     "fg": "#FFFFFF", "emoji": "🟠", "glow": "rgba(255,140,66,0.3)"},
    "Medium":   {"bg": COLORS["medium"],   "fg": "#1a1a2e", "emoji": "🟡", "glow": "rgba(255,217,61,0.3)"},
    "Low":      {"bg": COLORS["low"],      "fg": "#FFFFFF", "emoji": "🟢", "glow": "rgba(107,203,119,0.3)"},
}

EXAMPLE_POSTS = [
    {
        "label": "🌀 Hurricane Emergency",
        "text": (
            "URGENT: Category 5 hurricane making landfall in Miami. "
            "Massive flooding, buildings collapsing. People trapped on "
            "rooftops. Need immediate rescue in downtown area!"
        ),
    },
    {
        "label": "🌊 Flood Report",
        "text": (
            "Water levels rising in Houston neighborhoods. Streets are "
            "impassable. Families evacuating to higher ground. Red Cross "
            "shelters are filling up fast."
        ),
    },
    {
        "label": "🔥 Wildfire Update",
        "text": (
            "Wildfire spreading near Sacramento suburbs. Air quality "
            "hazardous. Fire crews working to contain the blaze. Residents "
            "advised to prepare for evacuation."
        ),
    },
    {
        "label": "🏔️ Earthquake Info",
        "text": (
            "Minor tremor felt in San Francisco area this morning. "
            "No structural damage reported. USGS recorded magnitude 3.2. "
            "Residents should review earthquake kits."
        ),
    },
]


@st.cache_resource(show_spinner=False)
def get_model():
    try:
        model = load_model()
        if model is not None:
            return model
    except Exception:
        pass
    try:
        sample_df = load_sample_data()
        if sample_df is not None and not sample_df.empty:
            return load_or_train_model(sample_df)
    except Exception:
        pass
    return None


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="section-header" style="font-size: 2rem;">
        <span class="accent-bar"></span>
        🔍 Analyze a Post
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color: #8888AA; margin-top: -0.5rem; margin-bottom: 1.5rem;">'
    "Paste any disaster-related text and get instant urgency classification "
    "with detailed NLP analysis — sentiment, entities, and preprocessed output.</p>",
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚡ Quick Examples")
    st.markdown(
        '<p style="color:#8888AA; font-size:0.82rem;">'
        "Click any example to populate the text box:</p>",
        unsafe_allow_html=True,
    )
    for example in EXAMPLE_POSTS:
        if st.button(example["label"], width="stretch"):
            st.session_state["analyze_input"] = example["text"]

    st.divider()
    st.markdown("#### Urgency Legend")
    for level, color in URGENCY_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{color};'
            f'display:inline-block;"></span>'
            f'<span style="color:#d0d0e0;font-size:0.85rem;">{level}</span></div>',
            unsafe_allow_html=True,
        )

# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading ML model..."):
    model = get_model()

if model is None:
    st.error(
        "Could not load the ML model. Ensure `data/sample_disaster_tweets.csv` "
        "exists and the utils/ package is set up."
    )
    st.stop()

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>Input</div>',
    unsafe_allow_html=True,
)

default_text = st.session_state.get("analyze_input", "")
user_text = st.text_area(
    "Paste a disaster-related social media post:",
    value=default_text,
    height=140,
    placeholder=(
        "Example: Hurricane approaching the coast. Massive evacuation "
        "underway. Need emergency shelters in Tampa immediately!"
    ),
)

analyze_clicked = st.button(
    "⚡ Analyze Post", type="primary", width="stretch"
)

# ── Analysis ───────────────────────────────────────────────────────────────────
if analyze_clicked and user_text.strip():
    st.divider()
    st.markdown(
        '<div class="section-header" style="font-size:1.6rem;">'
        '<span class="accent-bar"></span>Analysis Results</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Classifying urgency..."):
        predicted_urgency, confidence = predict_urgency(user_text, model)

    # Get per-class probabilities
    probabilities = {}
    try:
        cleaned = preprocess_text(user_text)
        if cleaned.strip():
            proba = model.predict_proba([cleaned])[0]
            classes = model.classes_
            probabilities = {cls: prob for cls, prob in zip(classes, proba)}
    except Exception:
        pass

    locations = extract_locations(user_text)
    sentiment = analyze_sentiment(user_text)
    preprocessed = preprocess_text(user_text)

    # ── Urgency badge + confidence ─────────────────────────────────────────
    badge_col, confidence_col = st.columns([1, 2])

    with badge_col:
        style = URGENCY_STYLES.get(predicted_urgency, URGENCY_STYLES["Medium"])
        st.markdown(
            f"""
            <div style="
                background: {style['bg']};
                color: {style['fg']};
                padding: 2rem;
                border-radius: 16px;
                text-align: center;
                font-size: 1.6rem;
                font-weight: 800;
                box-shadow: 0 8px 30px {style['glow']};
                animation: fadeIn 0.5s ease;
            ">
                {style['emoji']} {predicted_urgency} Urgency
            </div>
            <style>
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: scale(0.95); }}
                    to   {{ opacity: 1; transform: scale(1); }}
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    with confidence_col:
        st.markdown("#### Confidence Score")
        conf_value = float(confidence) if confidence else 0.0
        if conf_value > 1.0:
            conf_value = conf_value / 100.0
        st.progress(min(conf_value, 1.0))
        st.markdown(f"**{conf_value:.1%}** confidence in this prediction")

        if probabilities:
            st.markdown("**Class Probabilities:**")
            prob_cols = st.columns(len(probabilities))
            for col, (label, prob) in zip(prob_cols, probabilities.items()):
                emoji = URGENCY_EMOJI.get(label, "⚪")
                color = URGENCY_COLORS.get(label, "#888")
                col.markdown(
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:1.4rem;">{emoji}</div>'
                    f'<div style="color:{color}; font-weight:700; font-size:1.1rem;">'
                    f'{float(prob):.1%}</div>'
                    f'<div style="color:#8888AA; font-size:0.78rem;">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Entities + Sentiment ───────────────────────────────────────────────
    entity_col, sentiment_col = st.columns(2)

    with entity_col:
        st.markdown(
            '<div class="section-header" style="font-size:1.1rem;">'
            '<span class="accent-bar"></span>Extracted Locations</div>',
            unsafe_allow_html=True,
        )
        if locations:
            for loc in locations:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;'
                    f'background:rgba(255,255,255,0.04);border-radius:10px;'
                    f'padding:8px 14px;margin-bottom:6px;">'
                    f'<span style="color:#4ECDC4;">📍</span>'
                    f'<span style="color:#e0e0f0;font-weight:500;">{loc}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No specific locations detected.")

        st.markdown(
            '<div class="section-header" style="font-size:1.1rem; margin-top:1.5rem;">'
            '<span class="accent-bar"></span>Disaster Keywords</div>',
            unsafe_allow_html=True,
        )
        disaster_keywords = [
            "hurricane", "earthquake", "flood", "wildfire", "fire",
            "tornado", "tsunami", "storm", "evacuation", "rescue",
            "emergency", "damage", "destroyed", "trapped", "shelter",
            "casualties", "missing", "stranded", "collapsed", "debris",
        ]
        text_lower = user_text.lower()
        found_keywords = [kw for kw in disaster_keywords if kw in text_lower]
        if found_keywords:
            kw_html = " ".join(
                f'<span style="background:rgba(255,107,53,0.15);color:#FF8C42;'
                f'padding:4px 10px;border-radius:20px;font-size:0.85rem;'
                f'font-weight:500;margin:2px;">{kw}</span>'
                for kw in found_keywords
            )
            st.markdown(kw_html, unsafe_allow_html=True)
        else:
            st.info("No standard disaster keywords detected.")

    with sentiment_col:
        st.markdown(
            '<div class="section-header" style="font-size:1.1rem;">'
            '<span class="accent-bar"></span>Sentiment Analysis</div>',
            unsafe_allow_html=True,
        )

        if isinstance(sentiment, dict):
            compound = sentiment.get("compound", 0.0)
            pos = sentiment.get("pos", 0.0)
            neg = sentiment.get("neg", 0.0)
            neu = sentiment.get("neu", 0.0)
        else:
            compound = float(sentiment) if sentiment else 0.0
            pos, neg, neu = 0.0, 0.0, 0.0

        if compound >= 0.05:
            sent_label, sent_color, sent_emoji = "Positive", "#6BCB77", "😊"
        elif compound <= -0.05:
            sent_label, sent_color, sent_emoji = "Negative", "#FF3B3B", "😟"
        else:
            sent_label, sent_color, sent_emoji = "Neutral", "#FFD93D", "😐"

        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center; padding:1.5rem;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">{sent_emoji}</div>
                <div style="color:{sent_color}; font-size:1.5rem; font-weight:800;">
                    {sent_label}
                </div>
                <div style="color:#8888AA; font-size:0.9rem; margin-top:4px;">
                    Compound: {compound:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if pos or neg or neu:
            st.markdown("")  # spacer
            s1, s2, s3 = st.columns(3)
            s1.metric("Positive", f"{pos:.1%}")
            s2.metric("Negative", f"{neg:.1%}")
            s3.metric("Neutral", f"{neu:.1%}")

    st.divider()

    # ── Preprocessed text ──────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header" style="font-size:1.1rem;">'
        '<span class="accent-bar"></span>Preprocessed Text</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#8888AA;font-size:0.85rem;">What the NLP pipeline '
        "produces before feeding into the ML model:</p>",
        unsafe_allow_html=True,
    )
    if isinstance(preprocessed, list):
        st.code(" ".join(preprocessed), language=None)
    else:
        st.code(str(preprocessed), language=None)
    st.caption(
        "Pipeline: lowercasing → URL/mention removal → tokenization → "
        "stopword removal → lemmatization"
    )

elif analyze_clicked:
    st.warning("Please enter some text to analyze!")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    "⚡ Batch Analysis</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#8888AA; font-size:0.9rem; margin-top:-0.5rem;">'
    "Analyze multiple posts at once. Paste one post per line or upload a text file.</p>",
    unsafe_allow_html=True,
)

batch_tab1, batch_tab2 = st.tabs(["✏️ Paste Text", "📄 Upload File"])

with batch_tab1:
    batch_text = st.text_area(
        "Paste multiple posts (one per line):",
        height=150,
        placeholder=(
            "URGENT: Building collapse in downtown LA. People trapped!\n"
            "Minor earthquake felt in SF. No damage reported.\n"
            "Hurricane approaching Florida coast. Evacuate immediately!"
        ),
        key="batch_input",
    )
    batch_analyze = st.button("⚡ Analyze Batch", width="stretch", key="batch_btn")

with batch_tab2:
    batch_file = st.file_uploader(
        "Upload a .txt file (one post per line):",
        type=["txt"],
        key="batch_file",
    )
    if batch_file is not None:
        batch_text = batch_file.read().decode("utf-8")
        batch_analyze = True
    else:
        if "batch_text" not in dir():
            batch_text = ""
        if "batch_analyze" not in dir():
            batch_analyze = False

if batch_analyze and batch_text and batch_text.strip():
    lines = [line.strip() for line in batch_text.strip().split("\n") if line.strip()]

    if lines:
        st.markdown(
            f'<div class="notification-banner">'
            f'<span class="icon">📋</span>'
            f'<span class="text">Analyzing <b>{len(lines)}</b> posts...</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        results = []
        progress = st.progress(0)
        for i, line in enumerate(lines):
            pred, conf = predict_urgency(line, model)
            sentiment_result = analyze_sentiment(line)
            compound = (
                sentiment_result.get("compound", 0.0)
                if isinstance(sentiment_result, dict)
                else float(sentiment_result) if sentiment_result else 0.0
            )
            results.append({
                "Post": line[:120] + ("..." if len(line) > 120 else ""),
                "Urgency": pred,
                "Confidence": f"{float(conf):.1%}",
                "Sentiment": f"{compound:.3f}",
            })
            progress.progress((i + 1) / len(lines))

        progress.empty()

        # Summary stats
        from collections import Counter
        urgency_counter = Counter(r["Urgency"] for r in results)
        sum_cols = st.columns(4)
        for col, level in zip(sum_cols, URGENCY_ORDER):
            count = urgency_counter.get(level, 0)
            emoji = URGENCY_EMOJI.get(level, "")
            color = URGENCY_COLORS.get(level, "#888")
            col.markdown(
                f'<div style="text-align:center; background:rgba(255,255,255,0.04); '
                f'border:1px solid rgba(255,255,255,0.08); border-radius:12px; '
                f'padding:1rem; border-left:3px solid {color};">'
                f'<div style="font-size:1.8rem; font-weight:900; color:{color};">'
                f'{count}</div>'
                f'<div style="color:#8888AA; font-size:0.8rem; text-transform:uppercase; '
                f'letter-spacing:0.05em;">{emoji} {level}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")  # spacer

        # Results table
        import pandas as pd
        results_df = pd.DataFrame(results)

        def style_batch_urgency(val):
            style_map = {
                "Critical": f"background-color: {COLORS['critical']}; color: white; font-weight: 700; border-radius: 6px; padding: 2px 8px;",
                "High":     f"background-color: {COLORS['high']}; color: white; font-weight: 700; border-radius: 6px; padding: 2px 8px;",
                "Medium":   f"background-color: {COLORS['medium']}; color: #1a1a2e; font-weight: 700; border-radius: 6px; padding: 2px 8px;",
                "Low":      f"background-color: {COLORS['low']}; color: white; font-weight: 700; border-radius: 6px; padding: 2px 8px;",
            }
            return style_map.get(val, "")

        styled_results = results_df.style.map(
            style_batch_urgency, subset=["Urgency"]
        )
        st.dataframe(styled_results, width="stretch", height=300)

        # Export
        csv_results = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Results as CSV",
            data=csv_results,
            file_name="disasterlens_batch_results.csv",
            mime="text/csv",
            width="stretch",
        )

st.divider()
footer()

