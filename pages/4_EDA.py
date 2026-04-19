# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# pages/4_EDA.py — Exploratory Data Analysis
#
# In-depth text analytics and visualization suite. Generates word clouds,
# bigram/trigram frequency charts, character and word length distributions,
# sentiment breakdowns by disaster type, urgency co-occurrence matrices,
# and interactive Plotly charts styled with the global design system.
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re
from collections import Counter

st.set_page_config(
    page_title="EDA | DisasterLens AI",
    page_icon="📈",
    layout="wide",
)

from utils.theme import (
    inject_css, sidebar_nav, footer, apply_plotly_theme, get_colors,
    URGENCY_COLORS, URGENCY_ORDER, URGENCY_EMOJI, COLORS,
)
from utils.data_loader import load_sample_data
from utils.nlp_pipeline import preprocess_text, analyze_sentiment

inject_css()
sidebar_nav()

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


@st.cache_data
def get_sample_data():
    return load_sample_data()


@st.cache_data(show_spinner=False)
def compute_word_frequencies(texts: list, top_n: int = 20):
    word_counts = Counter()
    for text in texts:
        processed = preprocess_text(str(text))
        tokens = processed if isinstance(processed, list) else str(processed).split()
        tokens = [t for t in tokens if len(t) > 2 and not t.isdigit()]
        word_counts.update(tokens)
    return word_counts.most_common(top_n)


@st.cache_data(show_spinner=False)
def compute_bigrams(texts: list, top_n: int = 15):
    bigram_counts = Counter()
    for text in texts:
        processed = preprocess_text(str(text))
        tokens = processed if isinstance(processed, list) else str(processed).split()
        tokens = [t for t in tokens if len(t) > 2 and not t.isdigit()]
        for i in range(len(tokens) - 1):
            bigram_counts[f"{tokens[i]} {tokens[i+1]}"] += 1
    return bigram_counts.most_common(top_n)


@st.cache_data(show_spinner=False)
def compute_sentiment_scores(df: pd.DataFrame):
    sentiments = []
    for text in df["text"]:
        result = analyze_sentiment(str(text))
        if isinstance(result, dict):
            sentiments.append(result.get("compound", 0.0))
        else:
            sentiments.append(float(result) if result else 0.0)
    df_copy = df.copy()
    df_copy["sentiment_compound"] = sentiments
    return df_copy


# ── Data loading ───────────────────────────────────────────────────────────────
if "disaster_data" in st.session_state:
    df = st.session_state["disaster_data"]
else:
    df = get_sample_data()
    if df is not None:
        st.session_state["disaster_data"] = df

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="section-header" style="font-size: 2rem;">
        <span class="accent-bar"></span>
        📈 Exploratory Data Analysis
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color: #8888AA; margin-top: -0.5rem; margin-bottom: 1.5rem;">'
    "Deep-dive into word frequencies, bigrams, word clouds, sentiment "
    "distributions, and text-length analysis.</p>",
    unsafe_allow_html=True,
)

if df is None or df.empty:
    st.warning("No data available. Please load data on the Dashboard page first.")
    st.stop()

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎛️ Analysis Controls")
    top_n_words = st.slider(
        "Top N words:", min_value=5, max_value=50, value=20, step=5,
    )
    top_n_bigrams = st.slider(
        "Top N bigrams:", min_value=5, max_value=30, value=15, step=5,
    )
    selected_urgency_eda = st.multiselect(
        "Filter by urgency:",
        options=URGENCY_ORDER,
        default=URGENCY_ORDER,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 1. WORD FREQUENCY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    "Word Frequency Analysis</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#8888AA;font-size:0.9rem;">'
    "Most common words per urgency level (after NLP preprocessing).</p>",
    unsafe_allow_html=True,
)

with st.spinner("Computing word frequencies..."):
    urgency_tabs = st.tabs(
        [f"{URGENCY_EMOJI.get(u, '')} {u}" for u in selected_urgency_eda]
    )
    for tab, urgency in zip(urgency_tabs, selected_urgency_eda):
        with tab:
            urgency_texts = df[df["urgency"] == urgency]["text"].tolist()
            if not urgency_texts:
                st.info(f"No posts with '{urgency}' urgency level.")
                continue
            word_freqs = compute_word_frequencies(urgency_texts, top_n_words)
            if word_freqs:
                freq_df = pd.DataFrame(word_freqs, columns=["Word", "Count"])
                fig = px.bar(
                    freq_df, x="Count", y="Word", orientation="h",
                    title=f"Top {top_n_words} Words — {urgency} Urgency",
                    color_discrete_sequence=[URGENCY_COLORS.get(urgency, "#888")],
                    text="Count",
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    height=max(400, top_n_words * 24),
                )
                fig.update_traces(
                    textposition="outside",
                    textfont=dict(color=COLORS["text"]),
                    marker_line_width=0,
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No words extracted after preprocessing.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 2. BIGRAM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    "Bigram Analysis by Disaster Type</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#8888AA;font-size:0.9rem;">'
    "Most common two-word pairs — phrases like 'search rescue' or 'flood water'.</p>",
    unsafe_allow_html=True,
)

disaster_types = df["disaster_type"].dropna().unique().tolist()

with st.spinner("Computing bigrams..."):
    bigram_tabs = st.tabs(disaster_types)
    for tab, dtype in zip(bigram_tabs, disaster_types):
        with tab:
            dtype_texts = df[df["disaster_type"] == dtype]["text"].tolist()
            if not dtype_texts:
                st.info(f"No posts for disaster type '{dtype}'.")
                continue
            bigrams = compute_bigrams(dtype_texts, top_n_bigrams)
            if bigrams:
                bigram_df = pd.DataFrame(bigrams, columns=["Bigram", "Count"])
                fig = px.bar(
                    bigram_df, x="Count", y="Bigram", orientation="h",
                    title=f"Top {top_n_bigrams} Bigrams — {dtype}",
                    text="Count",
                    color_discrete_sequence=[COLORS["accent2"]],
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    height=max(380, top_n_bigrams * 26),
                )
                fig.update_traces(
                    textposition="outside",
                    textfont=dict(color=COLORS["text"]),
                    marker_line_width=0,
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No bigrams extracted after preprocessing.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 3. WORD CLOUDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    "Word Clouds</div>",
    unsafe_allow_html=True,
)

if WORDCLOUD_AVAILABLE:
    st.markdown(
        '<p style="color:#8888AA;font-size:0.9rem;">'
        "Visual word clouds — word size reflects frequency.</p>",
        unsafe_allow_html=True,
    )

    wc_cols = st.columns(2)
    col_idx = 0

    colormap_map = {
        "Critical": "Reds",
        "High": "Oranges",
        "Medium": "YlOrBr",
        "Low": "Greens",
    }

    for urgency in selected_urgency_eda:
        urgency_texts = df[df["urgency"] == urgency]["text"].tolist()
        if not urgency_texts:
            continue

        all_tokens = []
        for text in urgency_texts:
            processed = preprocess_text(str(text))
            if isinstance(processed, list):
                all_tokens.extend(processed)
            else:
                all_tokens.extend(str(processed).split())

        all_tokens = [t for t in all_tokens if len(t) > 2 and not t.isdigit()]
        combined_text = " ".join(all_tokens)
        if not combined_text.strip():
            continue

        with wc_cols[col_idx % 2]:
            st.markdown(
                f'<div style="color:{URGENCY_COLORS.get(urgency, "#888")}; '
                f'font-weight:700; font-size:1rem; margin-bottom:6px;">'
                f'{URGENCY_EMOJI.get(urgency, "")} {urgency} Urgency</div>',
                unsafe_allow_html=True,
            )

            wc = WordCloud(
                width=700, height=350,
                background_color="#0a0a1a",
                colormap=colormap_map.get(urgency, "viridis"),
                max_words=100,
                prefer_horizontal=0.7,
            ).generate(combined_text)

            fig_wc, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wc.patch.set_facecolor("#0a0a1a")
            fig_wc.tight_layout(pad=0)
            st.pyplot(fig_wc)
            plt.close(fig_wc)

        col_idx += 1
else:
    st.info(
        "The `wordcloud` library is not installed. "
        "Install with: `pip install wordcloud`"
    )
    for urgency in selected_urgency_eda:
        urgency_texts = df[df["urgency"] == urgency]["text"].tolist()
        if not urgency_texts:
            continue
        word_freqs = compute_word_frequencies(urgency_texts, top_n_words)
        if word_freqs:
            freq_df = pd.DataFrame(word_freqs, columns=["Word", "Count"])
            fig = px.bar(
                freq_df, x="Word", y="Count",
                title=f"Word Cloud Fallback — {urgency} Urgency",
                color_discrete_sequence=[URGENCY_COLORS.get(urgency, "#888")],
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 4. SENTIMENT DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    "Sentiment Distribution by Urgency</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#8888AA;font-size:0.9rem;">'
    "Compound sentiment scores (-1 negative to +1 positive) per urgency level.</p>",
    unsafe_allow_html=True,
)

with st.spinner("Computing sentiment scores..."):
    df_with_sentiment = compute_sentiment_scores(df)

sentiment_df = df_with_sentiment[
    df_with_sentiment["urgency"].isin(selected_urgency_eda)
]

if not sentiment_df.empty:
    fig_sentiment = px.box(
        sentiment_df,
        x="urgency", y="sentiment_compound",
        color="urgency",
        color_discrete_map=URGENCY_COLORS,
        category_orders={"urgency": URGENCY_ORDER},
        title="Sentiment Score Distribution by Urgency Level",
        labels={
            "urgency": "Urgency Level",
            "sentiment_compound": "Compound Sentiment Score",
        },
    )
    fig_sentiment.update_layout(showlegend=False, height=480)
    fig_sentiment.update_traces(
        marker=dict(opacity=0.7),
        line=dict(width=2),
    )
    apply_plotly_theme(fig_sentiment)
    st.plotly_chart(fig_sentiment, width="stretch")

    with st.expander("📊 Sentiment Summary Statistics"):
        summary = (
            sentiment_df.groupby("urgency")["sentiment_compound"]
            .describe()
            .round(3)
        )
        st.dataframe(summary, width="stretch")
else:
    st.info("No data for selected urgency levels.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 5. TEXT LENGTH DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    "Text Length Distribution</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#8888AA;font-size:0.9rem;">'
    "How long are posts at each urgency level? Longer posts may contain "
    "more detail while short urgent messages may indicate distress.</p>",
    unsafe_allow_html=True,
)

df_lengths = df[df["urgency"].isin(selected_urgency_eda)].copy()
df_lengths["text_length"] = df_lengths["text"].astype(str).apply(len)

if not df_lengths.empty:
    fig_length = px.histogram(
        df_lengths,
        x="text_length", color="urgency",
        color_discrete_map=URGENCY_COLORS,
        category_orders={"urgency": URGENCY_ORDER},
        title="Post Length (characters) by Urgency Level",
        labels={"text_length": "Text Length (characters)", "urgency": "Urgency"},
        barmode="overlay", opacity=0.7, nbins=30,
    )
    fig_length.update_layout(height=450)
    fig_length.update_traces(marker_line_width=0)
    apply_plotly_theme(fig_length)
    st.plotly_chart(fig_length, width="stretch")

    avg_cols = st.columns(len(selected_urgency_eda))
    for col, urgency in zip(avg_cols, selected_urgency_eda):
        subset = df_lengths[df_lengths["urgency"] == urgency]
        if not subset.empty:
            avg_len = int(subset["text_length"].mean())
            col.metric(f"Avg ({urgency})", f"{avg_len} chars")
else:
    st.info("No data for selected urgency levels.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 6. SENTIMENT vs TEXT LENGTH SCATTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    "Sentiment vs Text Length</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#8888AA;font-size:0.9rem;">'
    "Do longer posts tend to be more negative? This scatter plot reveals "
    "the relationship between text length and sentiment by urgency level.</p>",
    unsafe_allow_html=True,
)

if not df_with_sentiment.empty:
    scatter_df = df_with_sentiment[
        df_with_sentiment["urgency"].isin(selected_urgency_eda)
    ].copy()
    scatter_df["text_length"] = scatter_df["text"].astype(str).apply(len)

    if not scatter_df.empty:
        fig_scatter = px.scatter(
            scatter_df,
            x="text_length",
            y="sentiment_compound",
            color="urgency",
            color_discrete_map=URGENCY_COLORS,
            category_orders={"urgency": URGENCY_ORDER},
            title="Sentiment vs Text Length by Urgency",
            labels={
                "text_length": "Text Length (characters)",
                "sentiment_compound": "Sentiment Score",
                "urgency": "Urgency",
            },
            opacity=0.7,
            hover_data=["disaster_type"],
        )
        fig_scatter.update_traces(marker=dict(size=8, line=dict(width=0)))
        fig_scatter.update_layout(height=500)
        # Add reference line at sentiment = 0
        fig_scatter.add_hline(
            y=0, line_dash="dash",
            line_color="rgba(255,255,255,0.15)",
            annotation_text="Neutral",
            annotation_position="right",
            annotation_font_color="#8888AA",
        )
        apply_plotly_theme(fig_scatter)
        st.plotly_chart(fig_scatter, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 7. HEATMAP — DISASTER TYPE × URGENCY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    "Urgency × Disaster Type Heatmap</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#8888AA;font-size:0.9rem;">'
    "Which disaster types generate the most critical or high-urgency posts? "
    "Darker cells indicate higher post counts.</p>",
    unsafe_allow_html=True,
)

heatmap_df = df[df["urgency"].isin(selected_urgency_eda)]
if not heatmap_df.empty:
    pivot = pd.crosstab(heatmap_df["disaster_type"], heatmap_df["urgency"])
    # Reorder columns
    ordered_cols = [u for u in URGENCY_ORDER if u in pivot.columns]
    pivot = pivot[ordered_cols]

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0, "#0a0a1a"],
            [0.25, "#1a0a00"],
            [0.5, "#FF6B35"],
            [0.75, "#FF8C42"],
            [1, "#FFD93D"],
        ],
        text=pivot.values,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        hoverongaps=False,
    ))
    fig_heatmap.update_layout(
        title="Post Count: Disaster Type × Urgency Level",
        height=400,
        xaxis_title="Urgency Level",
        yaxis_title="Disaster Type",
    )
    apply_plotly_theme(fig_heatmap)
    st.plotly_chart(fig_heatmap, width="stretch")
else:
    st.info("No data for selected urgency levels.")

st.divider()
footer()

