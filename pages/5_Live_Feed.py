# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# pages/5_Live_Feed.py — Real-Time Social Media Feed
#
# Aggregates live disaster posts from Reddit, Bluesky, and Twitter/X.
# Each post is classified by urgency using the active ML model (DistilBERT
# or TF-IDF fallback). Features urgency-organized tabs, sort controls,
# summary metrics, source badges, and CSV export of classified data.
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Live Feed | DisasterLens AI",
    page_icon="📡",
    layout="wide",
)

from utils.theme import (
    inject_css, sidebar_nav, footer, get_colors,
    URGENCY_COLORS, URGENCY_ORDER, URGENCY_EMOJI,
)
from utils.classifier import get_active_classifier, predict_with_classifier
from utils.nlp_pipeline import analyze_sentiment

inject_css()
sidebar_nav()
c = get_colors()

# ── Source icons ───────────────────────────────────────────────────────────────
SOURCE_ICONS = {
    "Reddit": "🟠",
    "Bluesky": "🦋",
    "Twitter/X": "🐦",
}

SOURCE_COLORS = {
    "Reddit": "#FF4500",
    "Bluesky": "#0085FF",
    "Twitter/X": "#1DA1F2",
}


@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load the active classifier (BERT or TF-IDF)."""
    return get_active_classifier()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_reddit():
    """Fetch Reddit posts with 5-minute cache."""
    try:
        from utils.feeds.reddit_feed import fetch_reddit_posts
        return fetch_reddit_posts(limit=30, time_filter="day")
    except Exception as e:
        st.warning(f"Reddit fetch error: {e}")
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_bluesky():
    """Fetch Bluesky posts with 5-minute cache."""
    try:
        from utils.feeds.bluesky_feed import fetch_bluesky_posts
        return fetch_bluesky_posts(limit_per_query=15)
    except Exception as e:
        st.warning(f"Bluesky fetch error: {e}")
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_twitter():
    """Fetch Twitter posts with 5-minute cache."""
    try:
        from utils.feeds.twitter_feed import fetch_twitter_posts
        return fetch_twitter_posts(max_results=30)
    except Exception as e:
        st.warning(f"Twitter fetch error: {e}")
        return []


def check_feed_status():
    """Check which feeds are configured."""
    status = {}
    try:
        from utils.feeds.reddit_feed import is_configured as reddit_ok
        status["Reddit"] = reddit_ok()
    except Exception:
        status["Reddit"] = False
    try:
        from utils.feeds.bluesky_feed import is_configured as bsky_ok
        status["Bluesky"] = bsky_ok()
    except Exception:
        status["Bluesky"] = False
    try:
        from utils.feeds.twitter_feed import is_configured as twitter_ok
        status["Twitter/X"] = twitter_ok()
    except Exception:
        status["Twitter/X"] = False
    return status


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="section-header" style="font-size: 2rem;">
        <span class="accent-bar"></span>
        📡 Live Social Feed
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color: {c["text_secondary"]}; margin-top: -0.5rem; margin-bottom: 1.5rem;">'
    "Real-time disaster posts from Reddit, Bluesky, and Twitter/X — "
    "each automatically classified by urgency using our ML model.</p>",
    unsafe_allow_html=True,
)

# ── Check feed configuration ──────────────────────────────────────────────────
feed_status = check_feed_status()
any_configured = any(feed_status.values())

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📡 Feed Controls")

    st.markdown(
        f'<p style="color:{c["text_secondary"]}; font-size:0.82rem;">'
        "Select which platforms to monitor:</p>",
        unsafe_allow_html=True,
    )

    # Source toggles with status
    active_sources = []
    for source, configured in feed_status.items():
        icon = SOURCE_ICONS.get(source, "")
        if configured:
            if st.checkbox(f"{icon} {source}", value=True, key=f"src_{source}"):
                active_sources.append(source)
        else:
            st.checkbox(
                f"{icon} {source} (not configured)",
                value=False,
                disabled=True,
                key=f"src_{source}",
            )

    st.divider()

    # Twitter status callout
    try:
        from utils.feeds.twitter_feed import get_status as twitter_status
        tw_status = twitter_status()
    except Exception:
        tw_status = "🔒 Not configured"

    st.markdown("#### 🐦 Twitter/X Status")
    st.markdown(
        f'<div style="color:{c["text_secondary"]}; font-size:0.82rem; '
        f'background:{c["card_bg"]}; border:1px solid {c["card_border"]}; '
        f'border-radius:10px; padding:10px;">{tw_status}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Urgency filter
    selected_urgencies = st.multiselect(
        "Filter by urgency:",
        options=URGENCY_ORDER,
        default=URGENCY_ORDER,
        key="feed_urgency_filter",
    )

    st.divider()

    if st.button("🔄 Refresh Feed", width="stretch"):
        st.cache_data.clear()
        st.rerun()


# ── No feeds configured state ─────────────────────────────────────────────────
if not any_configured:
    st.markdown(
        f"""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📡</div>
            <h3 style="color: {c['text']}; margin-bottom: 0.5rem;">No Feeds Configured</h3>
            <p style="color: {c['text_secondary']}; max-width: 600px; margin: 0 auto 1.5rem;">
                To use the live feed, add your API credentials to
                <code>.streamlit/secrets.toml</code>. See
                <code>.streamlit/secrets.example.toml</code> for a template.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")  # spacer

    with st.expander("📋 How to configure feeds", expanded=True):
        st.markdown(
            f"""
**Reddit** (free):
1. Go to [reddit.com/prefs/apps](https://reddit.com/prefs/apps)
2. Create a "script" type application
3. Copy `client_id` and `client_secret` into secrets

**Bluesky** (free):
1. Go to [bsky.app/settings/app-passwords](https://bsky.app/settings/app-passwords)
2. Generate an app password
3. Add your handle and app password to secrets

**Twitter/X** (paid API):
1. Apply at [developer.twitter.com](https://developer.twitter.com)
2. Create a project and generate a Bearer Token
3. Add to secrets with `enabled = true`
            """
        )

    st.divider()
    footer()
    st.stop()


# ── Fetch posts from active sources ──────────────────────────────────────────
all_posts = []

with st.spinner("Fetching live posts from social media..."):
    if "Reddit" in active_sources:
        all_posts.extend(fetch_reddit())
    if "Bluesky" in active_sources:
        all_posts.extend(fetch_bluesky())
    if "Twitter/X" in active_sources:
        all_posts.extend(fetch_twitter())


if not all_posts:
    st.info("No posts found from the selected sources. Try refreshing or adjusting filters.")
    st.divider()
    footer()
    st.stop()


# ── Classify all posts ────────────────────────────────────────────────────────
with st.spinner("Classifying urgency levels..."):
    model_info = load_classifier()

    for post in all_posts:
        try:
            label, confidence = predict_with_classifier(post["text"], model_info)
            post["urgency"] = label
            post["confidence"] = confidence

            sent = analyze_sentiment(post["text"])
            post["sentiment"] = sent.get("compound", 0.0) if isinstance(sent, dict) else 0.0
        except Exception:
            post["urgency"] = "Medium"
            post["confidence"] = 0.0
            post["sentiment"] = 0.0


# ── Filter by urgency ─────────────────────────────────────────────────────────
filtered_posts = [p for p in all_posts if p["urgency"] in selected_urgencies]

# ── Feed status bar ───────────────────────────────────────────────────────────
source_counts = {}
for post in all_posts:
    src = post.get("source", "Unknown")
    source_counts[src] = source_counts.get(src, 0) + 1

status_parts = " + ".join(
    f'{SOURCE_ICONS.get(src, "")} {count} {src}'
    for src, count in source_counts.items()
)

st.markdown(
    f"""
    <div style="
        display: flex; align-items: center; gap: 10px;
        background: rgba(107,203,119,0.08);
        border: 1px solid rgba(107,203,119,0.2);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 1rem;
    ">
        <span class="pulse-dot"></span>
        <span style="color: #6BCB77; font-weight: 600;">Live Feed Active</span>
        <span style="color: {c['text_secondary']};">—</span>
        <span style="color: {c['text']};">
            {status_parts} = <b>{len(all_posts)} total</b>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Summary metrics ───────────────────────────────────────────────────────────
urgency_counts = {}
for post in filtered_posts:
    urg = post.get("urgency", "Medium")
    urgency_counts[urg] = urgency_counts.get(urg, 0) + 1

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Posts", len(filtered_posts))
m2.metric("🔴 Critical", urgency_counts.get("Critical", 0))
m3.metric("🟠 High", urgency_counts.get("High", 0))
m4.metric("🟡 Medium", urgency_counts.get("Medium", 0))
m5.metric("🟢 Low", urgency_counts.get("Low", 0))

# ── Model indicator ───────────────────────────────────────────────────────────
model_type = model_info[1] if model_info else "Unknown"
st.markdown(
    f'<div style="text-align:right; color:{c["text_secondary"]}; font-size:0.78rem; '
    f'margin-top:-0.5rem; margin-bottom:1rem;">'
    f'Classifier: <b>{model_type}</b></div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Post cards — organized by urgency tabs ─────────────────────────────────────
st.markdown(
    f'<div class="section-header"><span class="accent-bar"></span>Feed</div>',
    unsafe_allow_html=True,
)

if not filtered_posts:
    st.info("No posts match the selected urgency filters.")
else:
    # Sort control
    sort_option = st.selectbox(
        "Sort by:",
        ["Urgency (Critical first)", "Urgency (Low first)", "Newest first", "Confidence (highest)"],
        index=0,
        key="feed_sort",
    )

    urgency_rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

    if sort_option == "Urgency (Critical first)":
        filtered_posts.sort(key=lambda p: urgency_rank.get(p.get("urgency", "Medium"), 2))
    elif sort_option == "Urgency (Low first)":
        filtered_posts.sort(key=lambda p: urgency_rank.get(p.get("urgency", "Medium"), 2), reverse=True)
    elif sort_option == "Newest first":
        filtered_posts.sort(key=lambda p: p.get("timestamp", ""), reverse=True)
    elif sort_option == "Confidence (highest)":
        filtered_posts.sort(key=lambda p: p.get("confidence", 0), reverse=True)

    # Build tabs: All + each urgency level
    tab_labels = ["📋 All"]
    for level in URGENCY_ORDER:
        count = urgency_counts.get(level, 0)
        if count > 0:
            tab_labels.append(f"{URGENCY_EMOJI.get(level, '')} {level} ({count})")

    tabs = st.tabs(tab_labels)

    def render_post_card(post):
        """Render a single post as a styled card."""
        urgency = post.get("urgency", "Medium")
        confidence = post.get("confidence", 0.0)
        source = post.get("source", "Unknown")
        source_color = SOURCE_COLORS.get(source, "#888")
        source_icon = SOURCE_ICONS.get(source, "")
        urgency_color = URGENCY_COLORS.get(urgency, "#888")
        emoji = URGENCY_EMOJI.get(urgency, "⚪")

        extra_meta = ""
        if source == "Reddit":
            sub = post.get("subreddit", "")
            score = post.get("score", 0)
            comments = post.get("num_comments", 0)
            extra_meta = f" · {sub} · ⬆️{score} · 💬{comments}"
        elif source == "Bluesky":
            likes = post.get("likes", 0)
            reposts = post.get("reposts", 0)
            extra_meta = f" · ❤️{likes} · 🔄{reposts}"
        elif source == "Twitter/X":
            likes = post.get("likes", 0)
            rts = post.get("retweets", 0)
            extra_meta = f" · ❤️{likes} · 🔁{rts}"

        glow_class = f"glow-card-{urgency.lower()}"

        st.markdown(
            f"""
            <div class="{glow_class}" style="margin-bottom: 12px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="color:{urgency_color}; font-weight:700; font-size:0.85rem;">
                            {emoji} {urgency}
                        </span>
                        <span style="color:{c['text_secondary']}; font-size:0.78rem;">
                            {confidence:.0%} conf
                        </span>
                    </div>
                    <div style="display:flex; align-items:center; gap:6px;">
                        <span style="
                            background:{source_color}22;
                            color:{source_color};
                            padding:3px 10px;
                            border-radius:20px;
                            font-size:0.75rem;
                            font-weight:600;
                        ">{source_icon} {source}</span>
                    </div>
                </div>
                <div style="color:{c['text']}; font-size:0.9rem; line-height:1.5; margin-bottom:8px;">
                    {post['text'][:400]}{'...' if len(post['text']) > 400 else ''}
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:{c['text_secondary']}; font-size:0.75rem;">
                        {post.get('author', '')} · {post.get('timestamp', '')}{extra_meta}
                    </span>
                    <a href="{post.get('url', '#')}" target="_blank" style="
                        color:{c['accent']};
                        font-size:0.75rem;
                        text-decoration:none;
                    ">View original →</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tab 0: All posts
    with tabs[0]:
        for post in filtered_posts[:100]:
            render_post_card(post)

    # Tabs 1+: Per-urgency
    tab_idx = 1
    for level in URGENCY_ORDER:
        level_posts = [p for p in filtered_posts if p.get("urgency") == level]
        if not level_posts:
            continue
        with tabs[tab_idx]:
            st.markdown(
                f'<p style="color:{c["text_secondary"]}; font-size:0.85rem; margin-bottom:1rem;">'
                f'Showing <b>{len(level_posts)}</b> posts classified as '
                f'<span style="color:{URGENCY_COLORS.get(level, "#888")}; font-weight:700;">'
                f'{level}</span> urgency.</p>',
                unsafe_allow_html=True,
            )
            for post in level_posts[:50]:
                render_post_card(post)
        tab_idx += 1

st.divider()

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="section-header"><span class="accent-bar"></span>Export</div>',
    unsafe_allow_html=True,
)

if filtered_posts:
    export_df = pd.DataFrame([
        {
            "text": p["text"],
            "urgency": p.get("urgency", ""),
            "confidence": round(p.get("confidence", 0), 3),
            "sentiment": round(p.get("sentiment", 0), 3),
            "source": p.get("source", ""),
            "author": p.get("author", ""),
            "timestamp": p.get("timestamp", ""),
            "url": p.get("url", ""),
        }
        for p in filtered_posts
    ])

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Classified Feed as CSV",
        data=csv_data,
        file_name=f"disasterlens_live_feed_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        width="stretch",
    )

st.divider()
footer()
