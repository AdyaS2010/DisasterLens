# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# pages/1_Dashboard.py — Analytics Dashboard
#
# Data upload and visualization dashboard. Displays urgency distribution
# charts, disaster type breakdowns, geographic heatmaps, temporal trends,
# and detailed data tables. Supports user-uploaded CSVs alongside the
# built-in sample dataset.
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Dashboard | DisasterLens AI",
    page_icon="📊",
    layout="wide",
)

from utils.theme import (
    inject_css, sidebar_nav, footer, apply_plotly_theme, get_colors,
    URGENCY_COLORS, URGENCY_ORDER, DISASTER_COLORS, COLORS,
)
from utils.data_loader import load_sample_data, load_uploaded_data

inject_css()
sidebar_nav()


@st.cache_data
def get_sample_data():
    return load_sample_data()


def color_urgency(val):
    """Style function for urgency cells — safe for any value."""
    if not isinstance(val, str):
        return ""
    color_map = {
        "Critical": f"background-color: {COLORS['critical']}; color: white; font-weight: 700",
        "High":     f"background-color: {COLORS['high']}; color: white; font-weight: 700",
        "Medium":   f"background-color: {COLORS['medium']}; color: #1a1a2e; font-weight: 700",
        "Low":      f"background-color: {COLORS['low']}; color: white; font-weight: 700",
    }
    return color_map.get(val, "")


def _find_column(df_cols, keywords):
    """Find the first column whose lowercase name contains any keyword."""
    lower_cols = {c.lower().strip(): c for c in df_cols}
    # Exact match first
    for kw in keywords:
        if kw in lower_cols:
            return lower_cols[kw]
    # Substring match
    for kw in keywords:
        for lc, orig in lower_cols.items():
            if kw in lc:
                return orig
    return None


def _infer_urgency_from_values(series):
    """Try to convert various urgency/severity values to standard labels."""
    def _map_val(v):
        if not isinstance(v, str):
            # Could be a numeric severity (1-4 or 0-3)
            try:
                n = int(float(v))
                if n <= 1:
                    return "Critical"
                elif n == 2:
                    return "High"
                elif n == 3:
                    return "Medium"
                else:
                    return "Low"
            except (ValueError, TypeError):
                return "Medium"

        v_lower = v.strip().lower()

        # Direct matches
        direct = {
            "critical": "Critical", "crit": "Critical", "severe": "Critical",
            "emergency": "Critical", "urgent": "Critical", "very high": "Critical",
            "high": "High", "important": "High", "significant": "High",
            "moderate": "Medium", "medium": "Medium", "med": "Medium",
            "normal": "Medium", "standard": "Medium",
            "low": "Low", "minor": "Low", "info": "Low", "minimal": "Low",
            "informational": "Low", "none": "Low",
        }
        if v_lower in direct:
            return direct[v_lower]

        # Substring fallback
        for kw, label in [("crit", "Critical"), ("sever", "Critical"),
                          ("emerg", "Critical"), ("urgen", "Critical"),
                          ("high", "High"), ("import", "High"),
                          ("med", "Medium"), ("moder", "Medium"),
                          ("low", "Low"), ("minor", "Low"), ("info", "Low")]:
            if kw in v_lower:
                return label

        return "Medium"  # safe default

    return series.apply(_map_val)


def _infer_disaster_type(series):
    """Standardize disaster type values."""
    def _map_val(v):
        if not isinstance(v, str):
            return "Unknown"
        v_lower = v.strip().lower()
        for kw, label in [
            ("hurricane", "Hurricane"), ("cyclone", "Hurricane"),
            ("typhoon", "Hurricane"), ("tropical", "Hurricane"),
            ("earthquake", "Earthquake"), ("quake", "Earthquake"),
            ("seismic", "Earthquake"),
            ("flood", "Flood"), ("inundation", "Flood"),
            ("fire", "Wildfire"), ("wildfire", "Wildfire"),
            ("blaze", "Wildfire"), ("burn", "Wildfire"),
            ("tornado", "Tornado"), ("twister", "Tornado"),
            ("tsunami", "Tsunami"), ("storm", "Storm"),
        ]:
            if kw in v_lower:
                return label
        return v.strip().title() if v.strip() else "Unknown"
    return series.apply(_map_val)


def normalize_columns(dataframe):
    """
    Smart column normalization — finds the right columns by keyword matching
    on column names, so it works with any CSV format.
    """
    df = dataframe.copy()
    cols = list(df.columns)

    # ── Find TEXT column ──
    text_col = _find_column(cols, [
        "text", "tweet_text", "tweet", "message", "content", "post",
        "body", "description", "comment", "status", "review",
    ])
    if text_col and text_col != "text":
        df = df.rename(columns={text_col: "text"})
    elif text_col is None:
        # Use the column with the longest average string length
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            avg_lens = {c: df[c].astype(str).str.len().mean() for c in str_cols}
            best = max(avg_lens, key=avg_lens.get)
            df = df.rename(columns={best: "text"})

    cols = list(df.columns)

    # ── Find URGENCY column ──
    urgency_col = _find_column(cols, [
        "urgency", "urgency_label", "priority", "severity", "label",
        "level", "class", "classification", "risk", "threat",
    ])
    if urgency_col and urgency_col != "urgency":
        df = df.rename(columns={urgency_col: "urgency"})
    if "urgency" in df.columns:
        df["urgency"] = _infer_urgency_from_values(df["urgency"])
    else:
        df["urgency"] = "Medium"

    cols = list(df.columns)

    # ── Find DISASTER TYPE column ──
    dtype_col = _find_column(cols, [
        "disaster_type", "crisis_type", "category", "event_type",
        "disaster", "type", "event", "hazard", "crisis",
    ])
    if dtype_col and dtype_col != "disaster_type":
        df = df.rename(columns={dtype_col: "disaster_type"})
    if "disaster_type" in df.columns:
        df["disaster_type"] = _infer_disaster_type(df["disaster_type"])
    else:
        df["disaster_type"] = "Unknown"

    cols = list(df.columns)

    # ── Find LOCATION column ──
    loc_col = _find_column(cols, [
        "location_mentioned", "location", "place", "city", "region",
        "area", "address", "geo", "country", "state",
    ])
    if loc_col and loc_col != "location_mentioned":
        df = df.rename(columns={loc_col: "location_mentioned"})
    if "location_mentioned" not in df.columns:
        df["location_mentioned"] = ""

    cols = list(df.columns)

    # ── Find USER ID column ──
    user_col = _find_column(cols, ["user_id", "userid", "user", "username", "handle", "author"])
    if user_col and user_col != "user_id":
        df = df.rename(columns={user_col: "user_id"})

    # ── Find TIMESTAMP column ──
    time_col = _find_column(cols, ["timestamp", "time", "date", "created_at", "datetime"])
    if time_col and time_col != "timestamp":
        df = df.rename(columns={time_col: "timestamp"})

    # ── Find RELEVANCE column ──
    rel_col = _find_column(cols, ["relevance", "relevant", "rel", "importance"])
    if rel_col and rel_col != "relevance_label":
        df = df.rename(columns={rel_col: "relevance_label"})

    # ── Find INFORMATIVE column ──
    info_col = _find_column(cols, ["informative", "info", "informativeness", "is_informative"])
    if info_col and info_col != "informative_label":
        df = df.rename(columns={info_col: "informative_label"})

    # ── Ensure ID ──
    if "id" not in df.columns:
        # Check if there is an existing id-like column
        id_col = _find_column(cols, ["id", "tweet_id", "post_id", "index", "tid", "pid"])
        if id_col and id_col != "id":
            df = df.rename(columns={id_col: "id"})
        else:
            df["id"] = range(1, len(df) + 1)

    return df


# ── Sidebar — Data Source ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload a CSV file", type=["csv"],
        help="Auto-detects CrisisAwareTweets, CrisisNLP, and similar formats.",
    )
    use_sample = st.button("⚡ Load Sample Data", width="stretch")

# ── Data loading ───────────────────────────────────────────────────────────────
df = None
if uploaded_file is not None:
    try:
        df = load_uploaded_data(uploaded_file)
        df = normalize_columns(df)
        st.session_state["disaster_data"] = df
        st.sidebar.success(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
elif use_sample:
    df = get_sample_data()
    st.session_state["disaster_data"] = df
    st.sidebar.success(f"✅ Loaded {len(df)} sample rows")
elif "disaster_data" in st.session_state:
    df = st.session_state["disaster_data"]
    st.sidebar.info(f"Using loaded data ({len(df)} rows)")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="section-header" style="font-size: 2rem;">
        <span class="accent-bar"></span>
        📊 Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color: #8888AA; margin-top: -0.5rem; margin-bottom: 1.5rem;">'
    "Upload disaster-related data or use the sample dataset to explore urgency "
    "distributions, disaster-type breakdowns, and more.</p>",
    unsafe_allow_html=True,
)

if df is None or df.empty:
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📂</div>
            <h3 style="color: #F0F0F5; margin-bottom: 0.5rem;">No Data Loaded</h3>
            <p style="color: #8888AA; max-width: 500px; margin: 0 auto 1.5rem;">
                Use the sidebar to upload a CSV or click <b>Load Sample Data</b>
                to get started with 200 pre-labeled disaster tweets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "**Supported CSV formats:**\n\n"
        "DisasterLens auto-detects these column mappings:\n\n"
        "| Your Column | Also Accepts |\n"
        "|---|---|\n"
        "| `text` | `tweet_text`, `tweet`, `message` |\n"
        "| `urgency` | `urgency_label`, `label` |\n"
        "| `disaster_type` | `crisis_type`, `category`, `event_type` |\n"
        "| `location_mentioned` | `location`, `place`, `city` |"
    )
    st.stop()

# ── Summary metrics ────────────────────────────────────────────────────────────
urgency_counts = df["urgency"].value_counts()
total_posts = len(df)
critical_count = int(urgency_counts.get("Critical", 0))
high_count = int(urgency_counts.get("High", 0))
medium_count = int(urgency_counts.get("Medium", 0))
low_count = int(urgency_counts.get("Low", 0))
unique_locations = (
    df["location_mentioned"].dropna().loc[lambda s: s.str.strip() != ""].nunique()
)
disaster_types_count = df["disaster_type"].nunique()

st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>Summary</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total Posts", f"{total_posts:,}")
m2.metric("🔴 Critical", critical_count)
m3.metric("🟠 High", high_count)
m4.metric("🟡 Medium", medium_count)
m5.metric("🟢 Low", low_count)
m6.metric("📍 Locations", unique_locations)

st.divider()

# ── Urgency distribution charts ────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>Urgency Distribution</div>',
    unsafe_allow_html=True,
)

chart_left, chart_right = st.columns(2)

urgency_df = (
    df["urgency"].value_counts().reindex(URGENCY_ORDER).dropna().reset_index()
)
urgency_df.columns = ["Urgency", "Count"]

with chart_left:
    fig_pie = px.pie(
        urgency_df, names="Urgency", values="Count",
        color="Urgency", color_discrete_map=URGENCY_COLORS,
        title="Urgency Proportions", hole=0.45,
    )
    fig_pie.update_traces(
        textinfo="percent+label",
        textfont=dict(size=13, color="white"),
        marker=dict(line=dict(color="#0a0a1a", width=2)),
    )
    apply_plotly_theme(fig_pie)
    st.plotly_chart(fig_pie, width="stretch")

with chart_right:
    fig_bar = px.bar(
        urgency_df, x="Urgency", y="Count",
        color="Urgency", color_discrete_map=URGENCY_COLORS,
        title="Urgency Counts", text="Count",
    )
    fig_bar.update_layout(showlegend=False)
    fig_bar.update_traces(
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
        marker_line_width=0,
    )
    apply_plotly_theme(fig_bar)
    st.plotly_chart(fig_bar, width="stretch")

st.divider()

# ── Disaster type breakdown ────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>Disaster Type Breakdown</div>',
    unsafe_allow_html=True,
)

dtype_df = df["disaster_type"].value_counts().reset_index()
dtype_df.columns = ["Disaster Type", "Count"]

fig_disaster = px.bar(
    dtype_df, x="Disaster Type", y="Count",
    color="Disaster Type", color_discrete_map=DISASTER_COLORS,
    title="Posts by Disaster Type", text="Count",
)
fig_disaster.update_traces(
    textposition="outside",
    textfont=dict(color=COLORS["text"]),
    marker_line_width=0,
)
fig_disaster.update_layout(showlegend=False)
apply_plotly_theme(fig_disaster)
st.plotly_chart(fig_disaster, width="stretch")

st.divider()

# ── Cross-tabulation ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>Urgency × Disaster Type</div>',
    unsafe_allow_html=True,
)

cross_df = df.groupby(["disaster_type", "urgency"]).size().reset_index(name="Count")
fig_cross = px.bar(
    cross_df, x="disaster_type", y="Count",
    color="urgency", color_discrete_map=URGENCY_COLORS,
    category_orders={"urgency": URGENCY_ORDER},
    title="Urgency Distribution Within Each Disaster Type",
    barmode="stack",
    labels={"disaster_type": "Disaster Type", "urgency": "Urgency"},
)
fig_cross.update_traces(marker_line_width=0)
apply_plotly_theme(fig_cross)
st.plotly_chart(fig_cross, width="stretch")

st.divider()

# ── Location breakdown ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>Top Locations</div>',
    unsafe_allow_html=True,
)

location_data = (
    df["location_mentioned"]
    .dropna()
    .loc[lambda s: s.str.strip() != ""]
    .value_counts()
    .head(15)
    .reset_index()
)
location_data.columns = ["Location", "Posts"]

if not location_data.empty:
    fig_loc = px.bar(
        location_data, x="Posts", y="Location", orientation="h",
        title="Most Mentioned Locations",
        color_discrete_sequence=[COLORS["accent2"]],
        text="Posts",
    )
    fig_loc.update_layout(yaxis=dict(autorange="reversed"), height=450)
    fig_loc.update_traces(
        textposition="outside", marker_line_width=0,
        textfont=dict(color=COLORS["text"]),
    )
    apply_plotly_theme(fig_loc)
    st.plotly_chart(fig_loc, width="stretch")

st.divider()

# ── Critical posts spotlight ───────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>'
    '🚨 Critical Posts Spotlight</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#8888AA; font-size:0.9rem; margin-top:-0.5rem;">'
    "Posts classified as Critical urgency — requiring immediate response.</p>",
    unsafe_allow_html=True,
)

critical_df = df[df["urgency"] == "Critical"]
if not critical_df.empty:
    for _, row in critical_df.head(5).iterrows():
        loc_text = row.get("location_mentioned", "")
        loc_display = f" · 📍 {loc_text}" if loc_text and str(loc_text).strip() else ""
        dtype_display = row.get("disaster_type", "Unknown")
        st.markdown(
            f"""
            <div class="glow-card-critical" style="margin-bottom: 10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="color:#FF3B3B; font-weight:700; font-size:0.85rem;">
                        🔴 CRITICAL · {dtype_display}{loc_display}
                    </span>
                </div>
                <div style="color:#d0d0e0; font-size:0.9rem; line-height:1.5;">
                    {str(row.get('text', ''))[:300]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if len(critical_df) > 5:
        st.caption(f"Showing 5 of {len(critical_df)} critical posts")
else:
    st.info("No critical posts found in the dataset.")

st.divider()

# ── Data table + Export ────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="accent-bar"></span>Data Table</div>',
    unsafe_allow_html=True,
)

table_col, export_col = st.columns([3, 1])
with export_col:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export CSV",
        data=csv_data,
        file_name="disasterlens_data.csv",
        mime="text/csv",
        width="stretch",
    )

with st.expander("🛠️ Table Settings & Full Dataset", expanded=False):
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    
    with ctrl_col1:
        # Which columns to show
        available_cols = list(df.columns)
        default_shown = [
            "id", "user_id", "timestamp", "location_mentioned", 
            "disaster_type", "relevance_label", "urgency", 
            "informative_label", "text"
        ]
        # Only suggest defaults that actually exist
        default_cols = [c for c in default_shown if c in available_cols]
        
        display_cols = st.multiselect(
            "Columns to display:",
            options=available_cols,
            default=default_cols,
            key="display_cols_selector"
        )
    
    with ctrl_col2:
        # Sorting
        sort_on = st.selectbox(
            "Sort by:",
            # Only allow sorting on displayed columns
            options=display_cols if display_cols else ["id"],
            index=0,
            key="table_sort_col"
        )
    
    with ctrl_col3:
        sort_order = st.radio(
            "Order:",
            options=["Ascending", "Descending"],
            horizontal=True,
            index=0,
            key="table_sort_order"
        )

    st.divider()

    # Filter by urgency
    filter_urgency = st.multiselect(
        "Filter table by urgency:",
        options=URGENCY_ORDER,
        default=URGENCY_ORDER,
        key="table_filter",
    )
    
    # ── Process Data ──
    filtered_df = df[df["urgency"].isin(filter_urgency)].copy()
    
    # Apply sorting
    if sort_on in filtered_df.columns:
        filtered_df = filtered_df.sort_values(
            by=sort_on, 
            ascending=(sort_order == "Ascending")
        )
    
    # Filter columns
    if display_cols:
        final_df = filtered_df[display_cols]
    else:
        final_df = filtered_df

    # ── Display ──
    try:
        # Only apply color if urgency is actually in the displayed columns
        if "urgency" in final_df.columns:
            styled_df = final_df.style.map(color_urgency, subset=["urgency"])
            st.dataframe(styled_df, width="stretch", height=500)
        else:
            st.dataframe(final_df, width="stretch", height=500)
    except Exception:
        st.dataframe(final_df, width="stretch", height=500)
    
    st.caption(f"Showing {len(final_df):,} of {len(df):,} rows")

st.divider()
footer()

