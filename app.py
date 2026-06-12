"""
BUS5003 – NLP Analytics Platform
Amazon AirTag Review Intelligence
Group 10 | La Trobe University
"""

import io
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocess      import preprocess_dataframe
from utils.sentiment       import analyze_dataframe
from utils.topics          import train_lda, get_topic_keywords, assign_dominant_topic
from utils.explainability  import train_classifier, get_shap_explanations

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AirTag Review Intelligence",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Auth ──────────────────────────────────────────────────────────────────────
_USERS = {
    "admin": {
        "password": "admin123",
        "role":     "Administrator",
        "name":     "Administrator",
    }
}

def show_login():
    st.markdown("""
    <style>
    .login-wrap {
        max-width: 420px; margin: 80px auto 0 auto;
        background: #1a202c; border: 1px solid #2d3748;
        border-radius: 16px; padding: 44px 40px 36px 40px;
    }
    .login-logo { font-size: 36px; text-align: center; margin-bottom: 6px; }
    .login-title { color: #e2e8f0; font-size: 1.5rem; font-weight: 700;
                   text-align: center; margin-bottom: 4px; }
    .login-sub   { color: #718096; font-size: 13px; text-align: center;
                   margin-bottom: 28px; }
    .login-badge {
        display: inline-block; background: rgba(99,179,237,0.15); color: #63b3ed;
        border: 1px solid rgba(99,179,237,0.3); border-radius: 20px;
        padding: 3px 14px; font-size: 11px; font-weight: 600;
        letter-spacing: 0.06em; text-transform: uppercase;
        margin: 0 auto 16px auto; display: block; width: fit-content;
    }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown("""
        <div class="login-wrap">
            <div class="login-logo">🔍</div>
            <div class="login-title">Review Intelligence</div>
            <div class="login-sub">BUS5003 · Amazon AirTag NLP Platform</div>
            <span class="login-badge">Secure Access</span>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("🔐  Sign In", use_container_width=True):
            user = _USERS.get(username)
            if user and user["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["current_user"]  = username
                st.session_state["user_role"]      = user["role"]
                st.session_state["user_name"]      = user["name"]
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

        st.markdown(
            "<p style='color:#4a5568; font-size:11px; text-align:center; margin-top:18px;'>"
            "BUS5003 · Group 10 · La Trobe University · Sem A 2026</p>",
            unsafe_allow_html=True,
        )

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    show_login()
    st.stop()

# ── Cached wrappers ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_sentiment(df_json):
    df = pd.read_json(io.StringIO(df_json), orient='split')
    return analyze_dataframe(df, text_col='cleaned_text')

@st.cache_data(show_spinner=False)
def cached_lda(token_lists_json, n_topics):
    return train_lda(json.loads(token_lists_json), n_topics)

@st.cache_data(show_spinner=False)
def cached_shap(_pipeline, sample_texts_tuple):
    return get_shap_explanations(_pipeline, list(sample_texts_tuple), top_n=3)

def load_csv_with_progress(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV with a live progress bar."""
    total_bytes = len(file_bytes)
    bar = st.progress(0.0, text="📂 Reading file…  0%")

    class TrackedBytesIO(io.BytesIO):
        def read(self, size=-1):
            data = super().read(size)
            pct = min(self.tell() / total_bytes, 1.0)
            mb  = self.tell() / 1_048_576
            bar.progress(pct, text=f"📂 Reading file…  {pct*100:.0f}%  ({mb:.1f} / {total_bytes/1_048_576:.1f} MB)")
            return data

    df = pd.read_csv(TrackedBytesIO(file_bytes))
    bar.progress(1.0, text=f"✅ Loaded {len(df):,} rows")
    return df

def run_preprocess_with_progress(df, text_col):
    """Preprocessing with live progress bar."""
    bar = st.progress(0.0, text="🧹 Starting preprocessing…")
    def on_progress(pct, msg):
        bar.progress(float(pct), text=msg)
    result = preprocess_dataframe(df, text_col, progress_callback=on_progress)
    bar.progress(1.0, text=f"✅ Preprocessing complete — {len(result):,} rows")
    return result

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Keep header visible so sidebar toggle works */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDeployButton"] { display: none; }
[data-testid="stDecoration"]   { display: none; }

.stApp { background: #0f1117; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero h1 { color: #f0f4ff; font-size: 2.2rem; font-weight: 700; margin: 0 0 10px 0; line-height: 1.2; }
.hero p  { color: #8892a4; font-size: 1rem; margin: 0; max-width: 600px; }

/* ── Step headers ── */
.step-header { display: flex; align-items: center; gap: 14px; margin: 36px 0 20px 0; }
.step-badge {
    background: linear-gradient(135deg, #2b6cb0, #2c5282);
    color: white; border-radius: 10px;
    width: 38px; height: 38px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px; flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(43,108,176,0.4);
}
.step-title { color: #e2e8f0; font-size: 1.25rem; font-weight: 600; margin: 0; }
.step-sub   { color: #718096; font-size: 0.82rem; margin: 2px 0 0 0; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 14px; margin: 16px 0; flex-wrap: wrap; }
.metric-card {
    background: #1a202c; border: 1px solid #2d3748;
    border-radius: 12px; padding: 20px 24px;
    flex: 1; min-width: 140px; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4a5568; }
.metric-card .label { color: #718096; font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px; }
.metric-card .value { color: #e2e8f0; font-size: 1.9rem; font-weight: 700; line-height: 1; }
.metric-card .delta { font-size: 12px; margin-top: 6px; }
.metric-card.green  { border-left: 3px solid #48bb78; }
.metric-card.yellow { border-left: 3px solid #ecc94b; }
.metric-card.red    { border-left: 3px solid #fc8181; }
.metric-card.blue   { border-left: 3px solid #63b3ed; }
.metric-card.orange { border-left: 3px solid #ed8936; }

.section-divider { border: none; border-top: 1px solid #2d3748; margin: 32px 0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #111827 !important; border-right: 1px solid #1f2937 !important; }
[data-testid="stSidebar"] * { color: #d1d5db !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1a202c; border: 2px dashed #2d3748;
    border-radius: 12px; padding: 8px; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #4a90d9; }

/* ── Expander ── */
[data-testid="stExpander"] { background: #1a202c !important; border: 1px solid #2d3748 !important; border-radius: 10px !important; }

/* ── SHAP chips ── */
.chip-pos {
    display: inline-block; background: rgba(72,187,120,0.15); color: #68d391;
    border: 1px solid rgba(72,187,120,0.3); border-radius: 20px;
    padding: 3px 12px; font-size: 13px; margin: 3px 4px; font-weight: 500;
}
.chip-neg {
    display: inline-block; background: rgba(252,129,129,0.15); color: #fc8181;
    border: 1px solid rgba(252,129,129,0.3); border-radius: 20px;
    padding: 3px 12px; font-size: 13px; margin: 3px 4px; font-weight: 500;
}
.review-card { background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; }
.review-text { color: #a0aec0; font-size: 13.5px; line-height: 1.6; margin-bottom: 10px; }
.review-label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Topic pills ── */
.keyword-pill {
    display: inline-block; background: rgba(99,179,237,0.1); color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.2); border-radius: 16px;
    padding: 3px 11px; font-size: 12px; margin: 2px 3px;
}
.topic-card { background: #1a202c; border: 1px solid #2d3748; border-radius: 12px; padding: 16px 20px; margin-bottom: 10px; }
.topic-num  { color: #4a90d9; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #276749, #2f855a) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 13px 32px !important; font-weight: 700 !important; font-size: 16px !important;
    width: 100% !important; transition: opacity 0.2s !important;
    box-shadow: 0 4px 14px rgba(39,103,73,0.4) !important; margin-top: 8px !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.stDownloadButton > button {
    background: linear-gradient(135deg, #2b6cb0, #2c5282) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 12px 28px !important; font-weight: 600 !important; font-size: 15px !important;
    width: 100%; transition: opacity 0.2s !important;
    box-shadow: 0 4px 14px rgba(43,108,176,0.35) !important;
}
.stDownloadButton > button:hover { opacity: 0.88 !important; }

/* ── API tab ── */
.api-card {
    background: #1a202c; border: 1px solid #2d3748;
    border-radius: 12px; padding: 24px 28px; margin-bottom: 16px;
}
.api-card h4 { color: #e2e8f0; font-size: 1rem; font-weight: 600; margin: 0 0 6px 0; }
.api-card p  { color: #718096; font-size: 13px; margin: 0 0 16px 0; }
.api-status-ok  { color: #48bb78; font-weight: 600; font-size: 13px; }
.api-status-err { color: #fc8181; font-weight: 600; font-size: 13px; }
.code-block {
    background: #0d1117; border: 1px solid #2d3748; border-radius: 8px;
    padding: 14px 16px; font-family: monospace; font-size: 12px;
    color: #63b3ed; line-height: 1.7; overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme ─────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(color="#a0aec0", family="Inter"),
    title_font    = dict(color="#e2e8f0", size=15),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#a0aec0")),
    xaxis         = dict(gridcolor="#2d3748", linecolor="#2d3748", tickfont=dict(color="#718096")),
    yaxis         = dict(gridcolor="#2d3748", linecolor="#2d3748", tickfont=dict(color="#718096")),
    margin        = dict(l=16, r=16, t=40, b=16),
)
COLORS = {"Positive": "#48bb78", "Neutral": "#ecc94b", "Negative": "#fc8181"}

def apply_layout(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    _uname = st.session_state.get("user_name", "User")
    _urole = st.session_state.get("user_role", "")
    st.markdown(f"""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-size:22px; font-weight:700; color:#e2e8f0; margin-bottom:4px;">🔍 NLP Analytics Platform</div>
        <div style="font-size:12px; color:#4a5568;">BUS5003 · ANH TU NGUYEN · 22025993 · SEM A 2026</div>
        <div style="margin-top:12px; padding:10px 12px; background:#111827; border:1px solid #2d3748; border-radius:8px;">
            <div style="font-size:11px; color:#4a5568; text-transform:uppercase; letter-spacing:0.05em;">Signed in as</div>
            <div style="font-size:13px; color:#e2e8f0; font-weight:600; margin-top:2px;">{_uname}</div>
            <div style="font-size:11px; color:#63b3ed;">{_urole}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚪  Sign Out", use_container_width=True):
        for key in ["authenticated", "current_user", "user_role", "user_name"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("##### ⚙️ Analysis Settings")
    n_topics = st.slider("LDA Topics", min_value=3, max_value=10, value=5,
                         help="Number of topics for LDA topic modelling")
    star_filter = st.multiselect(
        "Star Rating Filter",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px; color:#4a5568; line-height:1.9;">
        <b style="color:#718096;">Pipeline</b><br>
        ① Upload CSV<br>
        ② Text Preprocessing<br>
        ③ Sentiment (VADER)<br>
        ④ Topic Modelling (LDA)<br>
        ⑤ SHAP Explainability<br>
        ⑥ Export Results
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛰 NLP Analytics Platform</div>
    <h1>Welcome to<br>Review Intelligence Platform!</h1>
    <p>Transform unstructured customer reviews into actionable business intelligence
    using Sentiment Analysis, Topic Modelling, and SHAP Explainability.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_analysis, tab_dashboard, tab_api = st.tabs(["📊 Analysis", "📈 Dashboard", "🔌 API Connection"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:

    # ── STEP 1: DATA UPLOAD ──────────────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <div class="step-badge">1</div>
        <div>
            <p class="step-title">Upload Review Data</p>
            <p class="step-sub">CSV file with a text column. Optional: rating / stars column for filtering.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your CSV here or click to browse",
        type=["csv"],
        label_visibility="collapsed",
    )

    if not uploaded_file:
        st.markdown("""
        <div style="background:#1a202c; border:1px solid #2d3748; border-radius:12px;
                    padding:20px 24px; color:#718096; font-size:14px; margin-top:12px;">
            👆 Upload a CSV to begin analysis. &nbsp;|&nbsp;
            No file? Use <code style="color:#63b3ed;">data/sample_reviews.csv</code> to try the demo.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    _file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if _file_key not in st.session_state:
        st.session_state[_file_key] = load_csv_with_progress(uploaded_file.read())
    df_raw = st.session_state[_file_key]

    text_cols = df_raw.select_dtypes(include='object').columns.tolist()
    if not text_cols:
        st.error("No text columns found in the uploaded file.")
        st.stop()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        text_col = st.selectbox("Review text column", text_cols)

    rating_col = None
    for candidate in ['rating', 'stars', 'score', 'Rating', 'Stars']:
        if candidate in df_raw.columns:
            rating_col = candidate
            break

    with col_right:
        if rating_col:
            df_raw[rating_col] = pd.to_numeric(df_raw[rating_col], errors='coerce')
            df_raw = df_raw[df_raw[rating_col].isin(star_filter)]

    n_rows = len(df_raw)
    n_cols = len(df_raw.columns)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card blue">
            <div class="label">Total Reviews</div>
            <div class="value">{n_rows:,}</div>
        </div>
        <div class="metric-card blue">
            <div class="label">Columns</div>
            <div class="value">{n_cols}</div>
        </div>
        <div class="metric-card blue">
            <div class="label">Text Column</div>
            <div class="value" style="font-size:1.1rem; padding-top:6px;">{text_col}</div>
        </div>
        <div class="metric-card blue">
            <div class="label">Rating Column</div>
            <div class="value" style="font-size:1.1rem; padding-top:6px;">{rating_col if rating_col else "—"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Preview raw data"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    # ── Run Analysis gate ────────────────────────────────────────────────────
    _run_key = f"ran_{_file_key}_{text_col}"
    if _run_key not in st.session_state:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center; color:#a0aec0; font-size:14px; margin-bottom:16px;">
            File loaded: <b style="color:#e2e8f0;">{n_rows:,} rows</b> ready.
            Click below to start the full NLP pipeline.
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶  Run Analysis"):
            st.session_state[_run_key] = True
            st.rerun()
        st.stop()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── STEP 2: PREPROCESSING ────────────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <div class="step-badge">2</div>
        <div>
            <p class="step-title">Text Preprocessing</p>
            <p class="step-sub">PII removal · lowercasing · stopword filtering · lemmatization</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _preprocess_key = f"preprocess_{_file_key}_{text_col}"
    if _preprocess_key not in st.session_state:
        st.session_state[_preprocess_key] = run_preprocess_with_progress(df_raw, text_col)
    else:
        st.markdown('<div style="color:#68d391; font-size:13px; margin-bottom:8px;">✅ Preprocessing cached</div>', unsafe_allow_html=True)
    df = st.session_state[_preprocess_key]

    avg_tokens = df['tokens'].apply(len).mean()
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card green">
            <div class="label">Reviews Processed</div>
            <div class="value">{len(df):,}</div>
        </div>
        <div class="metric-card green">
            <div class="label">Avg Tokens / Review</div>
            <div class="value">{avg_tokens:.1f}</div>
        </div>
        <div class="metric-card green">
            <div class="label">PII Masking</div>
            <div class="value" style="font-size:1.3rem; padding-top:4px;">✅ Applied</div>
        </div>
        <div class="metric-card green">
            <div class="label">Lemmatization</div>
            <div class="value" style="font-size:1.3rem; padding-top:4px;">✅ Done</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 View preprocessed samples"):
        st.dataframe(df[[text_col, 'cleaned_text', 'token_str']].head(5), use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── STEP 3: SENTIMENT ANALYSIS ───────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <div class="step-badge">3</div>
        <div>
            <p class="step-title">Sentiment Analysis</p>
            <p class="step-sub">VADER compound scoring · confidence bands · high-risk flagging</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running VADER sentiment analysis…"):
        df = cached_sentiment(df.to_json(orient='split'))

    counts     = df['label'].value_counts()
    pos_count  = counts.get('Positive', 0)
    neu_count  = counts.get('Neutral',  0)
    neg_count  = counts.get('Negative', 0)
    risk_count = int(df['high_risk'].sum())
    pos_pct    = pos_count / len(df) * 100 if len(df) else 0

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card green">
            <div class="label">🟢 Positive</div>
            <div class="value">{pos_count:,}</div>
            <div class="delta" style="color:#68d391;">{pos_pct:.1f}% of reviews</div>
        </div>
        <div class="metric-card yellow">
            <div class="label">🟡 Neutral</div>
            <div class="value">{neu_count:,}</div>
        </div>
        <div class="metric-card red">
            <div class="label">🔴 Negative</div>
            <div class="value">{neg_count:,}</div>
        </div>
        <div class="metric-card orange">
            <div class="label">⚠️ High-Risk</div>
            <div class="value">{risk_count:,}</div>
            <div class="delta" style="color:#ed8936;">Confidence &lt; 0.20</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(df, names='label', title="Sentiment Distribution",
                         color='label', color_discrete_map=COLORS, hole=0.50)
        fig_pie.update_traces(textposition='outside', textinfo='percent+label', textfont_color='#a0aec0')
        st.plotly_chart(apply_layout(fig_pie), use_container_width=True)

    with col_b:
        fig_hist = px.histogram(df, x='compound', nbins=40, title="Compound Score Distribution",
                                color_discrete_sequence=['#4a90d9'], labels={'compound': 'Compound Score'})
        fig_hist.add_vline(x= 0.05, line_dash='dash', line_color='#48bb78', annotation_text='Positive', annotation_font_color='#48bb78')
        fig_hist.add_vline(x=-0.05, line_dash='dash', line_color='#fc8181', annotation_text='Negative', annotation_font_color='#fc8181')
        st.plotly_chart(apply_layout(fig_hist), use_container_width=True)

    if rating_col:
        fig_bar = px.bar(
            df.groupby([rating_col, 'label']).size().reset_index(name='count'),
            x=rating_col, y='count', color='label', barmode='group',
            title="Sentiment by Star Rating", color_discrete_map=COLORS,
            labels={rating_col: 'Star Rating', 'count': 'Reviews'},
        )
        st.plotly_chart(apply_layout(fig_bar), use_container_width=True)

    if risk_count:
        st.markdown("""
        <div style="color:#ed8936; font-weight:600; font-size:14px; margin: 16px 0 8px 0;">
            ⚠️ High-Risk Reviews — Require Human Review
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df[df['high_risk']][[text_col, 'label', 'compound', 'confidence']].head(10), use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── STEP 4: TOPIC MODELLING ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="step-header">
        <div class="step-badge">4</div>
        <div>
            <p class="step-title">Topic Modelling (LDA)</p>
            <p class="step-sub">Latent Dirichlet Allocation · {n_topics} topics · TF-IDF features</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"Training LDA with {n_topics} topics…"):
        lda_model, vectorizer, doc_term_matrix, coherence = cached_lda(
            json.dumps(df['tokens'].tolist()), n_topics
        )
        topics = get_topic_keywords(lda_model, vectorizer)
        df['dominant_topic'] = assign_dominant_topic(lda_model, doc_term_matrix)
        # Persist fully-enriched df so the Dashboard tab can access it
        st.session_state[f"results_{_file_key}_{text_col}"] = df

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card blue">
            <div class="label">Topics Discovered</div>
            <div class="value">{n_topics}</div>
        </div>
        <div class="metric-card blue">
            <div class="label">Coherence Score</div>
            <div class="value">{coherence:.4f}</div>
            <div class="delta" style="color:#63b3ed;">0.50–0.70 is good</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.markdown('<div style="color:#e2e8f0; font-weight:600; margin-bottom:12px;">🏷 Top Keywords per Topic</div>', unsafe_allow_html=True)
        for topic_name, words in topics.items():
            pills = "".join(f'<span class="keyword-pill">{w}</span>' for w in words)
            st.markdown(f'<div class="topic-card"><div class="topic-num">{topic_name}</div><div>{pills}</div></div>', unsafe_allow_html=True)

    with col_t2:
        topic_counts = df['dominant_topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Reviews']
        topic_counts['Topic'] = 'Topic ' + topic_counts['Topic'].astype(str)
        fig_topics = px.bar(topic_counts.sort_values('Topic'), x='Topic', y='Reviews',
                            title="Reviews per Dominant Topic", color='Reviews',
                            color_continuous_scale=[[0, '#2c5282'], [1, '#63b3ed']], text='Reviews')
        fig_topics.update_traces(textposition='outside', textfont_color='#a0aec0')
        fig_topics.update_coloraxes(showscale=False)
        st.plotly_chart(apply_layout(fig_topics), use_container_width=True)

    cross = df.groupby(['dominant_topic', 'label']).size().reset_index(name='count')
    cross['dominant_topic'] = 'Topic ' + cross['dominant_topic'].astype(str)
    fig_cross = px.bar(cross, x='dominant_topic', y='count', color='label', barmode='stack',
                       title="Sentiment Distribution per Topic", color_discrete_map=COLORS,
                       labels={'dominant_topic': 'Topic', 'count': 'Reviews'})
    st.plotly_chart(apply_layout(fig_cross), use_container_width=True)


    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── STEP 5: SHAP EXPLAINABILITY ──────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <div class="step-badge">5</div>
        <div>
            <p class="step-title">SHAP Explainability</p>
            <p class="step-sub">Word-level contribution scores for high-risk reviews</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("Training classifier & computing SHAP values..."):
            pipeline, _ = train_classifier(df)
            sample_texts = df[df['high_risk']]['cleaned_text'].head(10).tolist()
    except ValueError as e:
        st.warning(f"SHAP explainability unavailable: {e}")
        sample_texts = []

    if sample_texts:
        explanations = cached_shap(pipeline, tuple(sample_texts))
        st.markdown(f"""
        <div style="color:#a0aec0; font-size:13.5px; margin-bottom:16px;">
            Showing <b style="color:#e2e8f0;">{len(explanations)}</b> high-risk review(s).
            <span style="color:#48bb78;">Green = positive driver</span>
            <span style="color:#fc8181;">Red = negative driver</span>
        </div>
        """, unsafe_allow_html=True)
        for exp in explanations:
            label_color = "#48bb78" if exp['predicted_label'] == 'Positive' else ("#fc8181" if exp['predicted_label'] == 'Negative' else "#ecc94b")
            chips = "".join(
                f'<span class="chip-pos">+{word} ({score:+.3f})</span>' if score > 0
                else f'<span class="chip-neg">{word} ({score:+.3f})</span>'
                for word, score in exp['top_words']
            )
            st.markdown(f"""
            <div class="review-card">
                <div class="review-text">"{exp['text'][:160]}"</div>
                <div style="margin-bottom:8px;"><span class="review-label" style="color:{label_color};">● {exp['predicted_label']}</span></div>
                <div>{chips}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high-risk reviews in the current selection.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── STEP 6: EXPORT ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <div class="step-badge">6</div>
        <div>
            <p class="step-title">Export Results</p>
            <p class="step-sub">Download the full enriched dataset as CSV</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    output_cols = [c for c in [text_col, 'cleaned_text', 'label', 'compound', 'confidence', 'high_risk', 'dominant_topic'] if c in df.columns]
    csv_bytes = df[output_cols].to_csv(index=False).encode('utf-8')

    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        st.download_button(label="Download Full Analysis (CSV)", data=csv_bytes,
                           file_name="nlp_analysis_results.csv", mime="text/csv")
    with col_info:
        exported = " | ".join(output_cols)
        st.markdown(f"""
        <div style="background:#1a202c; border:1px solid #2d3748; border-radius:10px;
                    padding:16px 20px; color:#718096; font-size:13px; line-height:1.8;">
            <b style="color:#e2e8f0;">Exported columns:</b> {exported}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; color:#2d3748; font-size:12px; margin-top:48px; padding:24px 0;">
        BUS5003 · Group 10 · NLP Analytics Platform · La Trobe University 2026
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 2 - DASHBOARD
# =============================================================================
with tab_dashboard:
    _dash_df_key = None
    for k in st.session_state:
        if k.startswith("results_"):
            _dash_df_key = k
    _has_results = _dash_df_key is not None

    if not _has_results:
        st.markdown("""
        <div style="background:#1a202c; border:1px dashed #2d3748; border-radius:16px;
                    padding:60px; text-align:center; margin-top:32px;">
            <div style="font-size:48px; margin-bottom:16px;">📈</div>
            <div style="color:#e2e8f0; font-size:1.2rem; font-weight:600; margin-bottom:8px;">No data yet</div>
            <div style="color:#718096; font-size:14px;">
                Run the full Analysis pipeline first (Steps 1-4) to populate this dashboard.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        dash_df = st.session_state[_dash_df_key]

        _counts   = dash_df['label'].value_counts() if 'label' in dash_df.columns else pd.Series(dtype=int)
        _pos      = int(_counts.get('Positive', 0))
        _neu      = int(_counts.get('Neutral',  0))
        _neg      = int(_counts.get('Negative', 0))
        _total    = len(dash_df)
        _risk     = int(dash_df['high_risk'].sum()) if 'high_risk' in dash_df.columns else 0
        _avg_comp = float(dash_df['compound'].mean()) if 'compound' in dash_df.columns else 0.0
        _pos_rate = _pos / _total * 100 if _total else 0
        _neg_rate = _neg / _total * 100 if _total else 0

        st.markdown(f"""
        <div style="color:#e2e8f0; font-size:1.4rem; font-weight:700; margin:8px 0 20px 0;">
            Executive Summary Dashboard
        </div>
        <div class="metric-row">
            <div class="metric-card blue">
                <div class="label">Total Reviews</div>
                <div class="value">{_total:,}</div>
            </div>
            <div class="metric-card green">
                <div class="label">Positive Rate</div>
                <div class="value">{_pos_rate:.1f}%</div>
                <div class="delta" style="color:#68d391;">{_pos:,} reviews</div>
            </div>
            <div class="metric-card red">
                <div class="label">Negative Rate</div>
                <div class="value">{_neg_rate:.1f}%</div>
                <div class="delta" style="color:#fc8181;">{_neg:,} reviews</div>
            </div>
            <div class="metric-card yellow">
                <div class="label">Avg Compound Score</div>
                <div class="value">{_avg_comp:.3f}</div>
                <div class="delta" style="color:#ecc94b;">Range -1 to +1</div>
            </div>
            <div class="metric-card orange">
                <div class="label">High-Risk Flagged</div>
                <div class="value">{_risk:,}</div>
                <div class="delta" style="color:#ed8936;">Need review</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_donut = px.pie(dash_df, names='label', title="Sentiment Breakdown",
                               color='label', color_discrete_map=COLORS, hole=0.60)
            fig_donut.update_traces(textposition='outside', textinfo='percent+label',
                                    textfont_color='#a0aec0')
            st.plotly_chart(apply_layout(fig_donut), use_container_width=True)

        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode   = "gauge+number+delta",
                value  = round(_avg_comp, 3),
                delta  = {'reference': 0, 'valueformat': '.3f'},
                title  = {'text': "Avg Sentiment Score", 'font': {'color': '#e2e8f0', 'size': 15}},
                gauge  = {
                    'axis':  {'range': [-1, 1], 'tickcolor': '#718096'},
                    'bar':   {'color': '#48bb78' if _avg_comp >= 0 else '#fc8181'},
                    'bgcolor': '#1a202c',
                    'steps': [
                        {'range': [-1, -0.05], 'color': 'rgba(252,129,129,0.15)'},
                        {'range': [-0.05, 0.05], 'color': 'rgba(236,201,75,0.10)'},
                        {'range': [0.05, 1],   'color': 'rgba(72,187,120,0.15)'},
                    ],
                },
                number = {'font': {'color': '#e2e8f0', 'size': 40}},
            ))
            fig_gauge.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig_gauge, use_container_width=True)

        if 'dominant_topic' in dash_df.columns:
            st.markdown('<div style="color:#e2e8f0; font-weight:600; font-size:1rem; margin:8px 0 12px 0;">Topic x Sentiment Heatmap</div>', unsafe_allow_html=True)
            heat_data = dash_df.groupby(['dominant_topic', 'label']).size().unstack(fill_value=0)
            heat_data.index = ['Topic ' + str(i) for i in heat_data.index]
            fig_heat = go.Figure(go.Heatmap(
                z=heat_data.values, x=heat_data.columns.tolist(), y=heat_data.index.tolist(),
                colorscale=[[0, '#0d1117'], [0.5, '#2b6cb0'], [1, '#63b3ed']],
                text=heat_data.values, texttemplate="%{text}",
                textfont={'color': '#e2e8f0', 'size': 12}, showscale=True,
            ))
            fig_heat.update_layout(**PLOT_LAYOUT, height=300)
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('<div style="color:#e2e8f0; font-weight:600; font-size:1rem; margin:16px 0 10px 0;">Most Negative Reviews (Bottom 10)</div>', unsafe_allow_html=True)
        if 'compound' in dash_df.columns:
            _known = {'cleaned_text', 'token_str', 'tokens', 'label', 'compound',
                      'confidence', 'high_risk', 'dominant_topic', 'label_num'}
            _text_candidates = [c for c in dash_df.columns if c not in _known]
            _show = (['compound', 'label', 'confidence'] +
                     ([_text_candidates[0]] if _text_candidates else []) +
                     (['dominant_topic'] if 'dominant_topic' in dash_df.columns else []))
            _show = [c for c in _show if c in dash_df.columns]
            st.dataframe(dash_df.nsmallest(10, 'compound')[_show], use_container_width=True)

        st.markdown("""
        <div style="text-align:center; color:#2d3748; font-size:12px; margin-top:48px; padding:24px 0;">
            BUS5003 · Group 10 · NLP Analytics Platform · La Trobe University 2026
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 3 - API DOCUMENTATION
# =============================================================================
with tab_api:
    st.markdown("""
<style>
.api-doc-header {
    background: linear-gradient(135deg, #1a1f2e, #0f3460);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px; padding: 32px 36px; margin-bottom: 28px;
}
.api-doc-title { color: #f0f4ff; font-size: 1.8rem; font-weight: 700; margin: 0 0 8px 0; }
.api-badge {
    display: inline-block; border-radius: 6px; padding: 2px 10px;
    font-size: 11px; font-weight: 700; margin-right: 6px; letter-spacing: 0.03em;
}
.badge-version { background: #4a5568; color: #e2e8f0; }
.badge-oas     { background: #276749; color: #9ae6b4; }
.badge-beta    { background: #744210; color: #fbd38d; }
.api-doc-section { background: #1a202c; border: 1px solid #2d3748; border-radius: 12px; padding: 24px 28px; margin-bottom: 16px; }
.api-doc-section h3 { color: #63b3ed; font-size: 1.05rem; font-weight: 700; margin: 0 0 10px 0; }
.api-doc-section p  { color: #a0aec0; font-size: 13px; margin: 0 0 8px 0; line-height: 1.6; }
.endpoint-group { background: #1a202c; border: 1px solid #2d3748; border-radius: 12px; margin-bottom: 12px; overflow: hidden; }
.endpoint-group-header { padding: 14px 20px; background: #161d2e; border-bottom: 1px solid #2d3748; color: #e2e8f0; font-weight: 600; font-size: 14px; }
.endpoint-group-desc { color: #718096; font-size: 12px; font-weight: 400; margin-left: 8px; }
.endpoint-row { display: flex; align-items: center; gap: 14px; padding: 11px 20px; border-bottom: 1px solid #1a202c; transition: background 0.15s; }
.endpoint-row:last-child { border-bottom: none; }
.endpoint-row:hover { background: #1e2533; }
.method-badge { display: inline-block; border-radius: 5px; padding: 3px 10px; font-size: 11px; font-weight: 800; min-width: 52px; text-align: center; letter-spacing: 0.04em; flex-shrink: 0; }
.method-get    { background: rgba(49,130,206,0.2);  color: #63b3ed; border: 1px solid rgba(49,130,206,0.4); }
.method-post   { background: rgba(39,103,73,0.25);  color: #68d391; border: 1px solid rgba(39,103,73,0.5); }
.method-put    { background: rgba(214,158,46,0.2);  color: #f6e05e; border: 1px solid rgba(214,158,46,0.4); }
.method-delete { background: rgba(197,48,48,0.2);   color: #fc8181; border: 1px solid rgba(197,48,48,0.4); }
.endpoint-path { color: #a0aec0; font-family: monospace; font-size: 13px; flex: 1; }
.endpoint-path span { color: #63b3ed; }
.endpoint-desc { color: #718096; font-size: 12px; flex: 2; }
.lock-icon     { color: #4a5568; font-size: 14px; flex-shrink: 0; }
.schema-pill { display: inline-block; background: #1a202c; border: 1px solid #2d3748; border-radius: 6px; padding: 4px 12px; font-size: 12px; color: #63b3ed; margin: 3px 4px; font-family: monospace; }
.rate-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
.rate-table th { color: #718096; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid #2d3748; padding: 8px 12px; text-align: left; }
.rate-table td { color: #e2e8f0; font-size: 13px; padding: 9px 12px; border-bottom: 1px solid #1a202c; }
.rate-table tr:last-child td { border-bottom: none; }
</style>

<div class="api-doc-header">
    <div class="api-doc-title">Review Intelligence API</div>
    <div style="margin: 8px 0 14px 0;">
        <span class="api-badge badge-version">v1.0</span>
        <span class="api-badge badge-oas">OAS3</span>
        <span class="api-badge badge-beta">BETA</span>
    </div>
    <p style="color:#8892a4; font-size:14px; margin:0; max-width:700px; line-height:1.6;">
        The Review Intelligence API is a RESTful API for ingesting customer review data and
        retrieving NLP analysis results including sentiment scoring, topic modelling, and
        SHAP-based explainability. Designed for integration with CX platforms, BI tools, and
        data warehouses.
        Base URL: <code style="color:#63b3ed;">https://api.review-intelligence.com</code>
    </p>
</div>
    """, unsafe_allow_html=True)

    col_ov, col_auth, col_rate = st.columns([2, 1, 1])

    with col_ov:
        st.markdown("""
        <div class="api-doc-section">
            <h3>Overview</h3>
            <p>Current capabilities include:</p>
            <ul style="color:#a0aec0; font-size:13px; margin:0; padding-left:18px; line-height:2;">
                <li>Upload and manage review datasets</li>
                <li>Trigger and retrieve sentiment analysis jobs (VADER)</li>
                <li>Run LDA topic modelling and retrieve topic keywords</li>
                <li>Fetch SHAP word-level explanations for reviews</li>
                <li>Export enriched datasets as CSV or JSON</li>
                <li>Register webhooks for real-time job completion events</li>
            </ul>
            <p style="margin-top:12px; font-size:12px; color:#4a5568;">
                Endpoints labelled BETA are intended for testing and may change without notice.
                Do not use in production integrations.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_auth:
        st.markdown("""
        <div class="api-doc-section">
            <h3>Authentication</h3>
            <p>All endpoints require a Bearer token in the <code style="color:#63b3ed;">Authorization</code> header.</p>
            <div class="code-block" style="margin-top:10px;">Authorization:<br>Bearer YOUR_API_KEY</div>
            <p style="margin-top:12px;">Keys are managed under <b style="color:#e2e8f0;">Admin &rarr; API Credentials</b>.</p>
            <p>Scopes: <code style="color:#63b3ed;">read</code>, <code style="color:#63b3ed;">write</code>, <code style="color:#63b3ed;">admin</code></p>
        </div>
        """, unsafe_allow_html=True)

    with col_rate:
        st.markdown("""
        <div class="api-doc-section">
            <h3>Rate Limits</h3>
            <p>Applied per API key. Custom plans available on request.</p>
            <table class="rate-table">
                <tr><th></th><th>Sandbox</th><th>Production</th></tr>
                <tr><td>Rate limit</td><td>1 req/s</td><td>10 req/s</td></tr>
                <tr><td>Burst</td><td>5</td><td>50</td></tr>
                <tr><td>Daily quota</td><td>500</td><td>50,000</td></tr>
            </table>
            <p style="margin-top:10px; font-size:12px; color:#4a5568;">
                Exceeding limits returns <code style="color:#fc8181;">429 Too Many Requests</code>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div style="color:#e2e8f0; font-weight:700; font-size:1.1rem; margin-bottom:16px;">Endpoint Reference</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="endpoint-group">
        <div class="endpoint-group-header">Authentication
            <span class="endpoint-group-desc">Obtain and refresh access tokens.</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-post">POST</span>
            <span class="endpoint-path">/oauth/token</span>
            <span class="endpoint-desc">Retrieve an access token using client credentials</span>
            <span class="lock-icon">&#x1F513;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-post">POST</span>
            <span class="endpoint-path">/oauth/token/refresh</span>
            <span class="endpoint-desc">Refresh an expired access token</span>
            <span class="lock-icon">&#x1F513;</span>
        </div>
    </div>

    <div class="endpoint-group">
        <div class="endpoint-group-header">Reviews
            <span class="endpoint-group-desc">Upload, retrieve, and manage review datasets.</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-get">GET</span>
            <span class="endpoint-path">/v1/reviews</span>
            <span class="endpoint-desc">Get paginated list of all uploaded review datasets</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-get">GET</span>
            <span class="endpoint-path">/v1/reviews/<span>{id}</span></span>
            <span class="endpoint-desc">Get a single review dataset by ID</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-post">POST</span>
            <span class="endpoint-path">/v1/reviews/upload</span>
            <span class="endpoint-desc">Upload a CSV or JSON file of reviews (multipart/form-data)</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-delete">DEL</span>
            <span class="endpoint-path">/v1/reviews/<span>{id}</span></span>
            <span class="endpoint-desc">Delete a review dataset by ID</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
    </div>

    <div class="endpoint-group">
        <div class="endpoint-group-header">Analysis
            <span class="endpoint-group-desc">Trigger NLP jobs and retrieve sentiment, topic, and explainability results.</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-post">POST</span>
            <span class="endpoint-path">/v1/analysis/sentiment</span>
            <span class="endpoint-desc">Trigger VADER sentiment analysis job on a dataset</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-get">GET</span>
            <span class="endpoint-path">/v1/analysis/sentiment/<span>{job_id}</span></span>
            <span class="endpoint-desc">Poll sentiment job status and retrieve results</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-post">POST</span>
            <span class="endpoint-path">/v1/analysis/topics</span>
            <span class="endpoint-desc">Trigger LDA topic modelling job — specify n_topics in body</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-get">GET</span>
            <span class="endpoint-path">/v1/analysis/topics/<span>{job_id}</span></span>
            <span class="endpoint-desc">Poll topic job and retrieve keywords + dominant topic assignments</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-post">POST</span>
            <span class="endpoint-path">/v1/analysis/explain</span>
            <span class="endpoint-desc">Compute SHAP word-level explanations for specified review IDs <span class="api-badge badge-beta">BETA</span></span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-get">GET</span>
            <span class="endpoint-path">/v1/analysis/explain/<span>{job_id}</span></span>
            <span class="endpoint-desc">Retrieve SHAP explanation results by job ID</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
    </div>

    <div class="endpoint-group">
        <div class="endpoint-group-header">Export
            <span class="endpoint-group-desc">Download enriched datasets and summary reports.</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-get">GET</span>
            <span class="endpoint-path">/v1/export/<span>{dataset_id}</span>/csv</span>
            <span class="endpoint-desc">Download full enriched dataset as CSV</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-get">GET</span>
            <span class="endpoint-path">/v1/export/<span>{dataset_id}</span>/json</span>
            <span class="endpoint-desc">Download enriched dataset as JSON array</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-get">GET</span>
            <span class="endpoint-path">/v1/export/<span>{dataset_id}</span>/summary</span>
            <span class="endpoint-desc">Get a JSON summary report with KPIs, sentiment counts, and top topics</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
    </div>

    <div class="endpoint-group">
        <div class="endpoint-group-header">Webhooks
            <span class="endpoint-group-desc">Register endpoints to receive real-time job completion notifications.</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-get">GET</span>
            <span class="endpoint-path">/v1/webhooks</span>
            <span class="endpoint-desc">List all registered webhook endpoints</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-post">POST</span>
            <span class="endpoint-path">/v1/webhooks</span>
            <span class="endpoint-desc">Register a new webhook URL for job completion events</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-put">PUT</span>
            <span class="endpoint-path">/v1/webhooks/<span>{id}</span></span>
            <span class="endpoint-desc">Update a registered webhook URL or event filter</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
        <div class="endpoint-row">
            <span class="method-badge method-delete">DEL</span>
            <span class="endpoint-path">/v1/webhooks/<span>{id}</span></span>
            <span class="endpoint-desc">Delete a registered webhook</span>
            <span class="lock-icon">&#x1F512;</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    schemas = ["ReviewUpload", "ReviewRecord", "SentimentJob", "SentimentResult",
               "TopicJob", "TopicResult", "TopicKeyword", "ExplainJob", "ExplainResult",
               "ShapWord", "ExportSummary", "Webhook", "OAuthToken", "Error",
               "PaginatedResponse", "JobStatus"]
    pills = "".join(f'<span class="schema-pill">{s}</span>' for s in schemas)
    st.markdown(f'<div style="color:#e2e8f0; font-weight:700; margin-bottom:10px;">Schemas</div>{pills}', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    ex1, ex2 = st.columns(2)
    with ex1:
        st.markdown('<div style="color:#718096; font-size:12px; margin-bottom:6px;">Get access token</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="code-block">
import requests<br><br>
r = requests.post(<br>
&nbsp;&nbsp;"https://api.review-intelligence.com/oauth/token",<br>
&nbsp;&nbsp;json={<br>
&nbsp;&nbsp;&nbsp;&nbsp;"client_id": "YOUR_CLIENT_ID",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"client_secret": "YOUR_SECRET",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"grant_type": "client_credentials"<br>
&nbsp;&nbsp;}<br>
)<br>
token = r.json()["access_token"]
        </div>
        """, unsafe_allow_html=True)

    with ex2:
        st.markdown('<div style="color:#718096; font-size:12px; margin-bottom:6px;">Upload reviews and trigger sentiment job</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="code-block">
headers = {"Authorization": f"Bearer {token}"}<br><br>
with open("reviews.csv", "rb") as f:<br>
&nbsp;&nbsp;upload = requests.post(<br>
&nbsp;&nbsp;&nbsp;&nbsp;"https://api.review-intelligence.com/v1/reviews/upload",<br>
&nbsp;&nbsp;&nbsp;&nbsp;headers=headers, files={"file": f}<br>
&nbsp;&nbsp;)<br>
dataset_id = upload.json()["dataset_id"]<br><br>
job = requests.post(<br>
&nbsp;&nbsp;"https://api.review-intelligence.com/v1/analysis/sentiment",<br>
&nbsp;&nbsp;headers=headers,<br>
&nbsp;&nbsp;json={"dataset_id": dataset_id, "text_column": "review_body"}<br>
)<br>
job_id = job.json()["job_id"]
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; color:#2d3748; font-size:12px; margin-top:48px; padding:24px 0;">
        BUS5003 · Group 10 · NLP Analytics Platform · La Trobe University 2026
    </div>
    """, unsafe_allow_html=True)
