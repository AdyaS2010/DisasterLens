# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# utils/theme.py — Premium Design System & UI Components
#
# Centralized design system powering the entire application. Generates
# dynamic CSS for light/dark themes, defines color tokens, urgency palettes,
# Plotly chart theming, accessibility settings (font scaling, high contrast,
# reduced motion), sidebar navigation, and reusable UI components.
# ──────────────────────────────────────────────────────────────────────────────

"""
DisasterLens AI — Premium Design System
Shared CSS, color tokens, Plotly layouts, and accessibility settings.
Supports light/dark mode toggle and accessibility preferences.
"""

import streamlit as st

# ── Color tokens ───────────────────────────────────────────────────────────────
COLORS_DARK = {
    "bg":            "#0a0a1a",
    "bg_secondary":  "#12122a",
    "surface":       "rgba(255,255,255,0.04)",
    "surface_hover": "rgba(255,255,255,0.08)",
    "glass":         "rgba(255,255,255,0.05)",
    "glass_border":  "rgba(255,255,255,0.10)",
    "accent":        "#6366F1",  # Indigo 500
    "accent2":       "#A855F7",  # Purple 500
    "text":          "#F0F0F5",
    "text_secondary": "#8888AA",
    "sidebar_bg":    "linear-gradient(180deg, #0d0d22 0%, #0a0a1a 100%)",
    "sidebar_text":  "#d0d0e0",
    "sidebar_muted": "#6a6a8a",
    "card_bg":       "rgba(255,255,255,0.04)",
    "card_border":   "rgba(255,255,255,0.08)",
    "divider":       "rgba(255,255,255,0.06)",
    "scrollbar_bg":  "#0a0a1a",
    "scrollbar_thumb": "rgba(255,255,255,0.15)",
    "main_bg":       "linear-gradient(160deg, #0a0a1a 0%, #0d0d24 40%, #12122a 100%)",
    "critical":      "#FF3B3B",
    "high":          "#FF8C42",
    "medium":        "#FFD93D",
    "low":           "#6BCB77",
}

COLORS_LIGHT = {
    "bg":            "#F5F6FA",
    "bg_secondary":  "#ECEDF3",
    "surface":       "rgba(0,0,0,0.03)",
    "surface_hover": "rgba(0,0,0,0.06)",
    "glass":         "rgba(255,255,255,0.7)",
    "glass_border":  "rgba(0,0,0,0.08)",
    "accent":        "#4F46E5",  # Indigo 600
    "accent2":       "#9333EA",  # Purple 600
    "text":          "#1a1a2e",
    "text_secondary": "#5a5a7a",
    "sidebar_bg":    "linear-gradient(180deg, #ECEDF3 0%, #F5F6FA 100%)",
    "sidebar_text":  "#2a2a3e",
    "sidebar_muted": "#7a7a9a",
    "card_bg":       "rgba(255,255,255,0.8)",
    "card_border":   "rgba(0,0,0,0.08)",
    "divider":       "rgba(0,0,0,0.08)",
    "scrollbar_bg":  "#F0F0F5",
    "scrollbar_thumb": "rgba(0,0,0,0.15)",
    "main_bg":       "linear-gradient(160deg, #F5F6FA 0%, #EEF0F7 40%, #E8EAF2 100%)",
    "critical":      "#FF3B3B",
    "high":          "#FF8C42",
    "medium":        "#FFD93D",
    "low":           "#6BCB77",
}

# Shared urgency colors (same in both themes)
URGENCY_COLORS = {
    "Critical": "#FF3B3B",
    "High":     "#FF8C42",
    "Medium":   "#FFD93D",
    "Low":      "#6BCB77",
}
URGENCY_ORDER = ["Critical", "High", "Medium", "Low"]

URGENCY_EMOJI = {
    "Critical": "🔴",
    "High":     "🟠",
    "Medium":   "🟡",
    "Low":      "🟢",
}

DISASTER_COLORS = {
    "Hurricane":  "#3B82F6",
    "Earthquake": "#EF4444",
    "Flood":      "#06B6D4",
    "Wildfire":   "#F59E0B",
}


def _get_theme():
    """Return which theme is active."""
    return st.session_state.get("theme", "dark")


def get_colors():
    """Return the active color dictionary."""
    return COLORS_DARK if _get_theme() == "dark" else COLORS_LIGHT


# Convenience aliases (default to dark, but pages should use get_colors())
COLORS = COLORS_DARK


# ── Plotly layout ──────────────────────────────────────────────────────────────
def get_plotly_layout():
    """Return Plotly layout dict for the active theme."""
    c = get_colors()
    is_dark = _get_theme() == "dark"
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=c["text"], size=13),
        title_font=dict(size=18, color=c["text"]),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["text_secondary"]),
        ),
        xaxis=dict(
            gridcolor=c["divider"],
            zerolinecolor=c["divider"],
        ),
        yaxis=dict(
            gridcolor=c["divider"],
            zerolinecolor=c["divider"],
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(
            bgcolor="#1a1a2e" if is_dark else "#ffffff",
            font_size=13,
            font_family="Inter, sans-serif",
        ),
    )

# Keep a static version for backward compatibility
PLOTLY_LAYOUT = get_plotly_layout() if "theme" not in st.session_state else get_plotly_layout()


def apply_plotly_theme(fig):
    """Apply the premium theme to a Plotly figure."""
    fig.update_layout(**get_plotly_layout())
    return fig


# ── CSS Generator ─────────────────────────────────────────────────────────────
def _build_css():
    """Build CSS for the active theme + accessibility settings."""
    c = get_colors()
    is_dark = _get_theme() == "dark"

    # Accessibility settings
    font_scale = st.session_state.get("font_scale", 1.0)
    high_contrast = st.session_state.get("high_contrast", False)
    reduced_motion = st.session_state.get("reduced_motion", False)

    base_font_size = round(16 * font_scale, 1)
    anim_duration = "0s" if reduced_motion else "0.3s"
    gradient_anim = "none" if reduced_motion else "gradient-shift 6s ease infinite"

    # High contrast overrides
    text_color = "#FFFFFF" if (is_dark and high_contrast) else ("#000000" if (not is_dark and high_contrast) else c["text"])
    text_sec = "#CCCCCC" if (is_dark and high_contrast) else ("#333333" if (not is_dark and high_contrast) else c["text_secondary"])
    card_border = "rgba(255,255,255,0.25)" if (is_dark and high_contrast) else ("rgba(0,0,0,0.2)" if (not is_dark and high_contrast) else c["card_border"])

    # Button colors adapt on light mode
    if is_dark:
        btn_color_1 = "#4338CA" # Deep Indigo
        btn_color_2 = "#312E81" # Very Deep Indigo
        btn_shadow = "rgba(49,46,129,0.3)"
        btn_shadow_hover = "rgba(49,46,129,0.5)"
        input_text = "#000000"  # Requested black text
        input_bg = "#FFFFFF"    # White background for visibility
    else:
        btn_color_1 = "#4F46E5" # Vibrant Indigo
        btn_color_2 = "#4338CA" # Solid Indigo
        btn_shadow = "rgba(79,70,229,0.15)"
        btn_shadow_hover = "rgba(79,70,229,0.25)"
        input_text = "#000000"  # Black text for light mode
        input_bg = "#FFFFFF"    # White background for light mode

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@800;900&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Themed Header ── */
[data-testid="stHeader"] {{
    background: {c['bg']} !important;
}}
[data-testid="stHeader"] svg, [data-testid="stHeader"] path {{
    fill: {text_color} !important;
    color: {text_color} !important;
}}

/* ── Hide Streamlit defaults ── */
#MainMenu {{ display: none !important; }}
footer {{ display: none !important; }}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], .main {{
    font-family: 'Inter', sans-serif !important;
    font-size: {base_font_size}px !important;
    color: {text_color} !important;
}}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {c['scrollbar_bg']}; }}
::-webkit-scrollbar-thumb {{ background: {c['scrollbar_thumb']}; border-radius: 3px; }}

/* ── Background ── */
[data-testid="stAppViewContainer"] {{
    background: {c['main_bg']} !important;
}}

/* ── GLOBAL TEXT COLOR OVERRIDES ── */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div {{
    color: {text_color};
}}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6 {{
    color: {text_color} !important;
}}
.stMarkdown p, .stMarkdown li, .stMarkdown span {{
    color: {text_color} !important;
}}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    color: {text_color} !important;
}}
.stMarkdown strong, .stMarkdown b {{
    color: {text_color} !important;
}}
[data-testid="stCaptionContainer"] {{
    color: {text_sec} !important;
}}
[data-testid="stWidgetLabel"] p {{
    color: {text_color} !important;
}}
[data-testid="stMarkdownContainer"] p {{
    color: {text_color} !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {c['sidebar_bg']} !important;
    border-right: 1px solid {c['divider']} !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {c['sidebar_text']} !important;
}}

/* ── Glass card ── */
.glass-card {{
    background: {c['card_bg']};
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 1.5rem;
    transition: all {anim_duration} cubic-bezier(0.4,0,0.2,1);
}}
.glass-card:hover {{
    background: {c['surface_hover']};
    border-color: {card_border};
    transform: translateY(-2px);
    box-shadow: 0 12px 40px {"rgba(0,0,0,0.4)" if is_dark else "rgba(0,0,0,0.08)"};
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {c['card_bg']};
    backdrop-filter: blur(16px);
    border: 1px solid {card_border};
    border-radius: 14px;
    padding: 1rem 1.2rem;
    transition: all {anim_duration} ease;
}}
[data-testid="stMetric"]:hover {{
    background: {c['surface_hover']};
    transform: translateY(-2px);
    box-shadow: 0 8px 30px {"rgba(0,0,0,0.3)" if is_dark else "rgba(0,0,0,0.06)"};
}}
[data-testid="stMetricLabel"] {{
    color: {text_sec} !important;
    font-weight: 500 !important;
    font-size: {round(0.85 * font_scale, 2)}rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricValue"] {{
    color: {text_color} !important;
    font-weight: 700 !important;
}}

/* ── Buttons ── */
.stButton>button {{
    background: linear-gradient(135deg, {btn_color_1} 0%, {btn_color_2} 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.6rem 1.5rem !important;
    transition: all {anim_duration} ease !important;
    box-shadow: 0 4px 15px {btn_shadow} !important;
}}
.stButton>button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px {btn_shadow_hover} !important;
}}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {{
    background: {input_bg} !important;
    border: 1px solid {card_border} !important;
    border-radius: 12px !important;
    color: {input_text} !important;
    -webkit-text-fill-color: {input_text} !important;
    font-family: 'Inter', sans-serif !important;
    transition: all {anim_duration} ease !important;
}}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {{
    color: {text_sec} !important;
    opacity: 0.7 !important;
}}
.stTextArea textarea:focus, .stTextInput input:focus {{
    border-color: {c['accent']} !important;
    background: {c['surface']} !important;
    box-shadow: 0 0 0 2px {"rgba(99,102,241,0.15)" if is_dark else "rgba(79,70,229,0.1)"} !important;
}}
|

/* ── Selects ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {c['card_bg']} !important;
    border: 1px solid {card_border} !important;
    border-radius: 12px !important;
    color: {text_color} !important;
}}
/* Dropdown menus */
[data-baseweb="popover"] {{
    background: {c['bg']} !important;
    border: 1px solid {card_border} !important;
    border-radius: 12px !important;
}}
[data-baseweb="menu"] {{
    background: {c['bg']} !important;
}}
[data-baseweb="menu"] li {{
    color: {text_color} !important;
}}
[data-baseweb="menu"] li:hover {{
    background: {c['surface_hover']} !important;
}}
/* Select slider */
.stSelectSlider p, .stSelectSlider span {{
    color: {text_color} !important;
}}
/* Checkbox / Radio labels */
.stCheckbox label span,
.stRadio label span {{
    color: {text_color} !important;
}}
/* Expander text */
[data-testid="stExpander"] summary span {{
    color: {text_color} !important;
}}
[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
    color: {text_color} !important;
}}
/* Download button */
.stDownloadButton > button {{
    background: {c['card_bg']} !important;
    border: 1px solid {card_border} !important;
    color: {text_color} !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}}
.stDownloadButton > button:hover {{
    background: {c['surface_hover']} !important;
    border-color: {c['accent']} !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: {"rgba(255,255,255,0.02)" if is_dark else "rgba(0,0,0,0.02)"};
    border-radius: 12px;
    padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important;
    color: {text_sec} !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}}
.stTabs [aria-selected="true"] {{
    background: {"rgba(255,107,53,0.15)" if is_dark else "rgba(232,93,44,0.1)"} !important;
    color: {c['accent']} !important;
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
    background: {c['card_bg']} !important;
    border-radius: 12px !important;
    color: {c['sidebar_text']} !important;
    font-weight: 500 !important;
}}

/* ── Divider ── */
hr {{
    border-color: {c['divider']} !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {c['surface']};
    border: 1px dashed {"rgba(255,255,255,0.12)" if is_dark else "rgba(0,0,0,0.12)"};
    border-radius: 14px;
    padding: 1rem;
}}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background: {c['accent']} !important;
}}

/* ── Dataframe ── */
.stDataFrame {{
    border-radius: 14px !important;
    overflow: hidden;
}}

/* ── Alerts ── */
.stAlert {{
    border-radius: 12px !important;
    border-left-width: 4px !important;
}}

/* ── Progress bar ── */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {c['accent']}, {URGENCY_COLORS['High']}) !important;
}}

/* ── Animated gradient text ── */
.gradient-text {{
    font-family: 'Outfit', 'Inter', sans-serif !important;
    font-size: {round(4.8 * font_scale, 1)}rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, {c['accent']}, {c['accent2']}, #3B82F6, #7E22CE);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: {gradient_anim};
    text-align: center;
    line-height: 1.1;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}}
@keyframes gradient-shift {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* ── Subtitle ── */
.subtitle {{
    font-size: {round(1.2 * font_scale, 2)}rem;
    text-align: center;
    color: {text_sec};
    margin-top: 0.3rem;
    margin-bottom: 2rem;
    font-weight: 400;
}}

/* ── Step / Feature cards ── */
.step-card {{
    background: {c['card_bg']};
    backdrop-filter: blur(12px);
    border: 1px solid {card_border};
    border-left: 4px solid {c['accent']};
    border-radius: 14px;
    padding: 1.3rem;
    margin-bottom: 0.6rem;
    color: {text_color};
    transition: all {anim_duration} cubic-bezier(0.4,0,0.2,1);
}}
.step-card:hover {{
    background: {c['surface_hover']};
    transform: translateY(-3px);
    box-shadow: 0 10px 30px {"rgba(0,0,0,0.3)" if is_dark else "rgba(0,0,0,0.06)"};
}}
.step-card h3 {{ margin: 0 0 0.5rem 0; color: {text_color}; font-weight: 700; font-size: {round(1.05 * font_scale, 2)}rem; }}
.step-card p  {{ margin: 0; color: {text_sec}; font-size: {round(0.92 * font_scale, 2)}rem; line-height: 1.5; }}

.feature-card {{
    background: {c['card_bg']};
    backdrop-filter: blur(12px);
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    transition: all {anim_duration} cubic-bezier(0.4,0,0.2,1);
}}
.feature-card:hover {{
    background: {"rgba(99,102,241,0.08)" if is_dark else "rgba(79,70,229,0.06)"};
    border-color: {"rgba(99,102,241,0.25)" if is_dark else "rgba(79,70,229,0.2)"};
    transform: translateY(-5px);
    box-shadow: 0 15px 40px {"rgba(99,102,241,0.15)" if is_dark else "rgba(79,70,229,0.08)"};
}}
.feature-icon {{ font-size: 2.8rem; margin-bottom: 0.8rem; }}
.feature-card h3 {{ color: {text_color}; font-weight: 700; margin: 0 0 0.5rem 0; font-size: {round(1.1 * font_scale, 2)}rem; }}
.feature-card p  {{ color: {text_sec}; font-size: {round(0.88 * font_scale, 2)}rem; line-height: 1.5; margin: 0; }}

/* ── Pulse animation ── */
.pulse-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #6BCB77;
    animation: {"pulse 2s ease-in-out infinite" if not reduced_motion else "none"};
    margin-right: 6px;
}}
@keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(107,203,119,0.6); }}
    70%  {{ box-shadow: 0 0 0 12px rgba(107,203,119,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(107,203,119,0); }}
}}

/* ── Section headers ── */
.section-header {{
    color: {text_color};
    font-weight: 700;
    font-size: {round(1.6 * font_scale, 1)}rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.section-header .accent-bar {{
    width: 4px; height: 24px;
    background: linear-gradient(180deg, {c['accent']}, {c['accent2']});
    border-radius: 2px;
}}

/* ── Footer ── */
.footer {{
    text-align: center;
    color: {c['sidebar_muted']};
    font-size: 0.85rem;
    padding: 1rem 0;
}}

/* ── Glow cards ── */
.glow-card-critical {{
    background: {"rgba(255,59,59,0.06)" if is_dark else "rgba(255,59,59,0.04)"};
    border: 1px solid {"rgba(255,59,59,0.15)" if is_dark else "rgba(255,59,59,0.12)"};
    border-radius: 14px; padding: 1.2rem;
    transition: all {anim_duration} ease;
}}
.glow-card-critical:hover {{ background: {"rgba(255,59,59,0.1)" if is_dark else "rgba(255,59,59,0.07)"}; }}
.glow-card-high {{
    background: {"rgba(255,140,66,0.06)" if is_dark else "rgba(255,140,66,0.04)"};
    border: 1px solid {"rgba(255,140,66,0.15)" if is_dark else "rgba(255,140,66,0.12)"};
    border-radius: 14px; padding: 1.2rem;
    transition: all {anim_duration} ease;
}}
.glow-card-medium {{
    background: {"rgba(255,217,61,0.06)" if is_dark else "rgba(255,217,61,0.04)"};
    border: 1px solid {"rgba(255,217,61,0.15)" if is_dark else "rgba(255,217,61,0.12)"};
    border-radius: 14px; padding: 1.2rem;
    transition: all {anim_duration} ease;
}}
.glow-card-low {{
    background: {"rgba(107,203,119,0.06)" if is_dark else "rgba(107,203,119,0.04)"};
    border: 1px solid {"rgba(107,203,119,0.15)" if is_dark else "rgba(107,203,119,0.12)"};
    border-radius: 14px; padding: 1.2rem;
    transition: all {anim_duration} ease;
}}

/* ── Code blocks ── */
.stCodeBlock, pre, code {{
    background: {c['surface']} !important;
    border: 1px solid {c['divider']} !important;
    border-radius: 12px !important;
}}

/* ── Tech badge ── */
.tech-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {c['card_bg']};
    border: 1px solid {card_border};
    border-radius: 50px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: {text_sec};
    transition: all {anim_duration} ease;
}}
.tech-badge:hover {{
    background: {"rgba(99,102,241,0.08)" if is_dark else "rgba(79,70,229,0.06)"};
    border-color: {"rgba(99,102,241,0.2)" if is_dark else "rgba(79,70,229,0.15)"};
    color: {c['accent']};
}}

/* ── Notification banner ── */
.notification-banner {{
    background: linear-gradient(135deg, {"rgba(99,102,241,0.1)" if is_dark else "rgba(79,70,229,0.06)"}, {"rgba(168,85,247,0.08)" if is_dark else "rgba(147,51,234,0.04)"});
    border: 1px solid {"rgba(99,102,241,0.2)" if is_dark else "rgba(79,70,229,0.12)"};
    border-radius: 14px;
    padding: 1rem 1.5rem;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 1rem;
}}
.notification-banner .icon {{ font-size: 1.5rem; }}
.notification-banner .text {{ color: {text_color}; font-size: 0.92rem; }}

/* ── Page links ── */
[data-testid="stPageLink"] {{
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}}
[data-testid="stPageLink"]:hover {{
    background: {"rgba(255,107,53,0.1)" if is_dark else "rgba(232,93,44,0.06)"} !important;
}}

/* ── Radio ── */
.stRadio > div {{
    background: {c['surface']};
    border-radius: 12px;
    padding: 8px;
}}

/* ── Fade-in animations ── */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.fade-in-up {{ animation: fadeInUp {anim_duration} ease forwards; }}
.fade-in-up-delay-1 {{ animation-delay: 0.1s; opacity: 0; }}
.fade-in-up-delay-2 {{ animation-delay: 0.2s; opacity: 0; }}
.fade-in-up-delay-3 {{ animation-delay: 0.3s; opacity: 0; }}
.fade-in-up-delay-4 {{ animation-delay: 0.4s; opacity: 0; }}
</style>
"""


def inject_css():
    """Inject the themed CSS into the current page."""
    st.markdown(_build_css(), unsafe_allow_html=True)


def sidebar_nav():
    """Render a clean, professional sidebar."""
    with st.sidebar:
        # Brand
        c = get_colors()
        is_dark = _get_theme() == "dark"

        st.markdown(
            f"""
            <div style="text-align:center; padding: 0.8rem 0 0.3rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.2rem;">🔍</div>
                <div style="
                    font-size: 1.3rem; font-weight: 800;
                    background: linear-gradient(135deg, {c['accent']}, #FF3B3B);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                ">DisasterLens AI</div>
                <div style="color: {c['sidebar_muted']}; font-size: 0.75rem; margin-top: 2px;">
                    Disaster Intelligence Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # Navigation (only our clean links, Streamlit's default nav is hidden via CSS)
        st.page_link("app.py", label="🏠  Home", width="stretch")
        st.page_link("pages/1_Dashboard.py", label="📊  Dashboard", width="stretch")
        st.page_link("pages/2_Live_Map.py", label="🗺️  Live Map", width="stretch")
        st.page_link("pages/3_Analyze_Post.py", label="🔍  Analyze Post", width="stretch")
        st.page_link("pages/4_EDA.py", label="📈  Exploratory Analysis", width="stretch")
        st.page_link("pages/5_Live_Feed.py", label="📡  Live Feed", width="stretch")
        st.divider()

        # ── Settings ──────────────────────────────────────────────────────
        with st.expander("⚙️ Settings", expanded=False):
            # Theme toggle
            theme_choice = st.selectbox(
                "Theme",
                options=["Dark", "Light"],
                index=0 if _get_theme() == "dark" else 1,
                key="theme_selector",
            )
            new_theme = "dark" if theme_choice == "Dark" else "light"
            if new_theme != st.session_state.get("theme", "dark"):
                st.session_state["theme"] = new_theme
                st.rerun()

            st.markdown("")  # spacer

            # Font size
            font_size = st.select_slider(
                "Text Size",
                options=[0.85, 0.9, 1.0, 1.1, 1.2, 1.3],
                value=st.session_state.get("font_scale", 1.0),
                format_func=lambda x: {0.85: "Small", 0.9: "Compact", 1.0: "Default", 1.1: "Large", 1.2: "X-Large", 1.3: "XX-Large"}[x],
                key="font_size_slider",
            )
            if font_size != st.session_state.get("font_scale", 1.0):
                st.session_state["font_scale"] = font_size
                st.rerun()

            # High contrast
            hc = st.checkbox(
                "High Contrast",
                value=st.session_state.get("high_contrast", False),
                key="hc_toggle",
            )
            if hc != st.session_state.get("high_contrast", False):
                st.session_state["high_contrast"] = hc
                st.rerun()

            # Reduced motion
            rm = st.checkbox(
                "Reduced Motion",
                value=st.session_state.get("reduced_motion", False),
                key="rm_toggle",
            )
            if rm != st.session_state.get("reduced_motion", False):
                st.session_state["reduced_motion"] = rm
                st.rerun()

        # Footer
        st.markdown(
            f"""
            <div style="color: {c['sidebar_muted']}; font-size: 0.72rem; line-height: 1.6; margin-top: 0.5rem;">
                Python · Streamlit · scikit-learn · spaCy<br>
                NLTK · Plotly · Folium
            </div>
            """,
            unsafe_allow_html=True,
        )


def footer():
    """Render a minimal branded footer."""
    c = get_colors()
    st.markdown(
        f'<p class="footer">DisasterLens AI · Disaster Intelligence Platform</p>',
        unsafe_allow_html=True,
    )
