import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(
    page_title="Live Map | DisasterLens AI",
    page_icon="🗺️",
    layout="wide",
)

from utils.theme import (
    inject_css, sidebar_nav, footer, get_colors,
    URGENCY_COLORS, URGENCY_ORDER, URGENCY_EMOJI, COLORS,
)
from utils.data_loader import load_sample_data
from utils.nlp_pipeline import extract_locations, geocode_location

inject_css()
sidebar_nav()

MARKER_COLORS = {
    "Critical": "red",
    "High":     "orange",
    "Medium":   "beige",
    "Low":      "green",
}

HEX_MARKER = {
    "Critical": "#FF3B3B",
    "High":     "#FF8C42",
    "Medium":   "#FFD93D",
    "Low":      "#6BCB77",
}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_usgs_earthquakes():
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [])
            if len(coords) < 2:
                continue
            mag = props.get("mag", 0) or 0
            place = props.get("place", "Unknown location")
            time_ms = props.get("time", 0)
            event_time = (
                datetime.fromtimestamp(time_ms / 1000).strftime("%Y-%m-%d %H:%M")
                if time_ms else "Unknown"
            )
            if mag >= 6.0:
                urgency = "Critical"
            elif mag >= 5.0:
                urgency = "High"
            elif mag >= 4.0:
                urgency = "Medium"
            else:
                urgency = "Low"
            results.append({
                "lat": coords[1], "lon": coords[0],
                "text": f"M{mag:.1f} earthquake — {place} ({event_time} UTC)",
                "urgency": urgency, "disaster_type": "Earthquake",
                "location": place,
            })
        return results
    except Exception as e:
        st.warning(f"Could not fetch USGS data: {e}")
        return []


@st.cache_data(ttl=600, show_spinner=False)
def fetch_nasa_eonet():
    try:
        url = "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&limit=50"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        category_urgency = {
            "wildfires": "High", "volcanoes": "Critical",
            "severeStorms": "High", "floods": "High",
            "earthquakes": "High", "landslides": "Medium",
            "seaLakeIce": "Low", "snow": "Low",
            "tempExtremes": "Medium", "drought": "Medium",
        }
        results = []
        for event in data.get("events", []):
            title = event.get("title", "Unknown event")
            categories = event.get("categories", [])
            cat_id = categories[0].get("id", "unknown") if categories else "unknown"
            cat_title = categories[0].get("title", "Natural Event") if categories else "Natural Event"
            urgency = category_urgency.get(cat_id, "Medium")
            geometries = event.get("geometry", [])
            if not geometries:
                continue
            latest_geo = geometries[-1]
            coords = latest_geo.get("coordinates", [])
            geo_date = latest_geo.get("date", "")[:10]
            if len(coords) >= 2:
                results.append({
                    "lat": coords[1], "lon": coords[0],
                    "text": f"{title} ({geo_date})",
                    "urgency": urgency, "disaster_type": cat_title,
                    "location": title,
                })
        return results
    except Exception as e:
        st.warning(f"Could not fetch NASA EONET data: {e}")
        return []


@st.cache_data(show_spinner=False)
def cached_geocode(location_text: str):
    try:
        return geocode_location(location_text)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def geocode_all_posts(df: pd.DataFrame):
    results = []
    for _, row in df.iterrows():
        location_str = None
        loc_mentioned = str(row.get("location_mentioned", "")).strip()
        if loc_mentioned and loc_mentioned.lower() not in ("", "nan", "none"):
            location_str = loc_mentioned
        else:
            text = str(row.get("text", ""))
            extracted = extract_locations(text)
            if extracted:
                location_str = extracted[0]
        if not location_str:
            continue
        coords = cached_geocode(location_str)
        if coords and len(coords) == 2:
            lat, lon = coords
            results.append({
                "lat": lat, "lon": lon,
                "text": str(row.get("text", ""))[:300],
                "urgency": row.get("urgency", "Medium"),
                "disaster_type": row.get("disaster_type", "Unknown"),
                "location": location_str,
            })
    return results


@st.cache_data
def get_sample_data():
    return load_sample_data()


# ── Ensure data ────────────────────────────────────────────────────────────────
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
        🗺️ Live Disaster Map
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color: #8888AA; margin-top: -0.5rem; margin-bottom: 1rem;">'
    "Real-time disaster events plotted on an interactive map. "
    "Markers are color-coded by urgency level.</p>",
    unsafe_allow_html=True,
)

if df is None or df.empty:
    st.warning("No data available. Please load data on the Dashboard page first.")
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🗺️ Map Controls")
    data_source = st.radio(
        "Data Source:",
        options=["🌐 Live Disasters (Real-time)", "📄 Sample Dataset"],
        index=0,
        help="Toggle between live feeds and sample tweets",
    )
    st.divider()
    selected_urgencies = st.multiselect(
        "Urgency Levels:",
        options=URGENCY_ORDER,
        default=URGENCY_ORDER,
        help="Filter markers by urgency",
    )
    st.divider()
    st.markdown("#### Legend")
    for level, color in URGENCY_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{color};'
            f'display:inline-block;"></span>'
            f'<span style="color:#d0d0e0;font-size:0.85rem;">{level}</span></div>',
            unsafe_allow_html=True,
        )

# ── Load data ──────────────────────────────────────────────────────────────────
if "Live" in data_source:
    with st.spinner("Fetching live disaster data from USGS & NASA..."):
        usgs_data = fetch_usgs_earthquakes()
        eonet_data = fetch_nasa_eonet()
        map_data = usgs_data + eonet_data

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
            <span style="color: #8888AA;">—</span>
            <span style="color: #d0d0e0;">
                {len(usgs_data)} earthquakes (USGS) +
                {len(eonet_data)} events (NASA EONET) =
                <b>{len(map_data)} total</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    with st.spinner("Geocoding sample data locations..."):
        progress_bar = st.progress(0, text="Preparing location data...")
        map_data = geocode_all_posts(df)
        progress_bar.progress(100, text="Geocoding complete!")

filtered_data = [p for p in map_data if p["urgency"] in selected_urgencies]

# ── Stats ──────────────────────────────────────────────────────────────────────
stat_cols = st.columns(4)
stat_cols[0].metric("Total Events", len(map_data))
stat_cols[1].metric("Visible", len(filtered_data))
stat_cols[2].metric("🔴 Critical", len([p for p in filtered_data if p["urgency"] == "Critical"]))
stat_cols[3].metric("🟠 High", len([p for p in filtered_data if p["urgency"] == "High"]))

if not map_data:
    st.warning("No posts could be geocoded. Location data may be missing.")
    st.stop()
if not filtered_data:
    st.info("No posts match the selected urgency filters.")
    st.stop()

st.divider()

# ── Build map ──────────────────────────────────────────────────────────────────
avg_lat = sum(p["lat"] for p in filtered_data) / len(filtered_data)
avg_lon = sum(p["lon"] for p in filtered_data) / len(filtered_data)

disaster_map = folium.Map(
    location=[avg_lat, avg_lon],
    zoom_start=3,
    tiles="CartoDB voyager",
)

for point in filtered_data:
    urgency = point["urgency"]
    color = HEX_MARKER.get(urgency, "#888888")
    radius = 12 if urgency == "Critical" else (9 if urgency == "High" else 7)
    opacity = 0.9 if urgency in ("Critical", "High") else 0.7

    popup_html = (
        f"<div style='width:280px; font-family:Inter,sans-serif; "
        f"background:#1a1a2e; color:#e0e0f0; padding:14px; "
        f"border-radius:12px; border-left:4px solid {color};'>"
        f"<div style='font-weight:700; color:{color}; "
        f"margin-bottom:6px; font-size:1rem;'>"
        f"{URGENCY_EMOJI.get(urgency, '')} {urgency} Urgency</div>"
        f"<div style='color:#a0a0c0; font-size:0.85rem; margin-bottom:4px;'>"
        f"<b>Type:</b> {point['disaster_type']}</div>"
        f"<div style='color:#a0a0c0; font-size:0.85rem; margin-bottom:8px;'>"
        f"<b>Location:</b> {point['location']}</div>"
        f"<hr style='border-color:rgba(255,255,255,0.1); margin:6px 0;'>"
        f"<div style='color:#c0c0d0; font-size:0.82rem; "
        f"font-style:italic; line-height:1.4;'>{point['text']}</div>"
        f"</div>"
    )

    folium.CircleMarker(
        location=[point["lat"], point["lon"]],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=opacity,
        weight=2,
        popup=folium.Popup(popup_html, max_width=320),
        tooltip=f"{urgency} — {point['location']}",
    ).add_to(disaster_map)

folium_static(disaster_map, width=1200, height=620)

with st.expander("📋 View geocoded data table"):
    geo_df = pd.DataFrame(filtered_data)
    st.dataframe(geo_df, width="stretch")

st.divider()
footer()
