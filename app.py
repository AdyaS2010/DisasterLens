import streamlit as st

st.set_page_config(
    page_title="DisasterLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.theme import inject_css, sidebar_nav, footer, get_colors

inject_css()
sidebar_nav()

c = get_colors()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="gradient-text">DisasterLens AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    "Real-time Disaster Intelligence · NLP-Powered Urgency Classification"
    "</p>",
    unsafe_allow_html=True,
)

# ── Live status ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="display:flex; justify-content:center; align-items:center;
        gap:8px; margin-bottom:2rem;">
        <span class="pulse-dot"></span>
        <span style="color:#6BCB77; font-weight:600; font-size:0.9rem;">
            System Online</span>
        <span style="color:{c['sidebar_muted']};">·</span>
        <span style="color:{c['text_secondary']}; font-size:0.85rem;">
            Live earthquake &amp; natural event feeds active</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Metrics ────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Disaster Types", "4", "Tracked")
col2.metric("Urgency Levels", "4", "Classified")
col3.metric("ML Pipeline", "TF-IDF + LR", "Trained")
col4.metric("Live Feeds", "USGS · NASA", "Connected")

st.divider()

# ── About ──────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown(
        '<div class="section-header"><span class="accent-bar"></span>'
        "What is DisasterLens AI?</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **DisasterLens AI** is a disaster response platform that uses
        **Natural Language Processing** and **Machine Learning** to analyze
        social media posts during natural disasters.

        It classifies the **urgency level** of each post so emergency
        responders can prioritize the most critical calls for help.

        - 🔎 **Analyze** post text via NLP techniques
        - ⚡ **Classify** urgency — Critical, High, Medium, or Low
        - 🗺️ **Map** locations on interactive maps
        - 📊 **Visualize** trends for coordinated response
        """
    )

with right:
    st.markdown(
        '<div class="section-header"><span class="accent-bar"></span>'
        "Urgency Levels</div>",
        unsafe_allow_html=True,
    )
    from utils.theme import URGENCY_COLORS as _UC
    levels = [
        ("🔴", "Critical", _UC["Critical"], "Immediate life-threatening danger"),
        ("🟠", "High", _UC["High"], "Significant risk, needs fast response"),
        ("🟡", "Medium", _UC["Medium"], "Important but not immediately dangerous"),
        ("🟢", "Low", _UC["Low"], "Informational, minimal urgency"),
    ]
    for emoji, level, color, desc in levels:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px;
                background:{c['card_bg']}; border-left:3px solid {color};
                border-radius:10px; padding:10px 14px; margin-bottom:6px;">
                <span style="font-size:1.3rem;">{emoji}</span>
                <div>
                    <div style="color:{color}; font-weight:700; font-size:0.9rem;">{level}</div>
                    <div style="color:{c['text_secondary']}; font-size:0.8rem;">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── How It Works ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    "How It Works</div>",
    unsafe_allow_html=True,
)

steps = [
    ("📥", "1. Ingest Data",
     "Upload a CSV of social media posts or use the built-in sample dataset of 200 labeled tweets."),
    ("🧹", "2. Preprocess",
     "Text is cleaned, tokenized, and transformed. Locations are extracted and sentiment is analyzed."),
    ("🤖", "3. Classify",
     "The trained ML model predicts urgency (Critical, High, Medium, Low) for each post."),
    ("📊", "4. Visualize",
     "Results are shown on dashboards, maps, and charts for actionable intelligence."),
]

cols = st.columns(4)
for col, (icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(
            f"""
            <div class="step-card">
                <h3>{icon} {title}</h3>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Pages ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    "Explore</div>",
    unsafe_allow_html=True,
)

features = [
    ("📊", "Dashboard", "Upload data, view urgency distributions and disaster breakdowns."),
    ("🗺️", "Live Map", "Real-time disasters on an interactive map with urgency markers."),
    ("🔍", "Analyze Post", "Paste text for instant urgency prediction with NLP analysis."),
    ("📈", "EDA", "Word clouds, bigrams, sentiment distributions, and more."),
]

cols = st.columns(4)
for col, (icon, name, desc) in zip(cols, features):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <h3>{name}</h3>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
footer()
