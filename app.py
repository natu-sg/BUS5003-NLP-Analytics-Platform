"""
BUS5003 – NLP Analytics Platform
Amazon AirTag Review Intelligence
Group 10 | La Trobe University
"""

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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Background ── */
.stApp {
    background: #0f1117;
}

/* ── Hero banner ── */
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
.hero h1 {
    color: #f0f4ff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 10px 0;
    line-height: 1.2;
}
.hero p {
    color: #8892a4;
    font-size: 1rem;
    margin: 0;
    max-width: 600px;
}

/* ── Step headers ── */
.step-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 36px 0 20px 0;
}
.step-badge {
    background: linear-gradient(135deg, #2b6cb0, #2c5282);
    color: white;
    border-radius: 10px;
    width: 38px; height: 38px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(43,108,176,0.4);
}
.step-title {
    color: #e2e8f0;
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
}
.step-sub {
    color: #718096;
    font-size: 0.82rem;
    margin: 2px 0 0 0;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 14px; margin: 16px 0; flex-wrap: wrap; }
.metric-card {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px 24px;
    flex: 1; min-width: 140px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4a5568; }
.metric-card .label {
    color: #718096; font-size: 11px;
    font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 8px;
}
.metric-card .value {
    color: #e2e8f0; font-size: 1.9rem;
    font-weight: 700; line-height: 1;
}
.metric-card .delta { font-size: 12px; margin-top: 6px; }
.metric-card.green  { border-left: 3px solid #48bb78; }
.metric-card.yellow { border-left: 3px solid #ecc94b; }
.metric-card.red    { border-left: 3px solid #fc8181; }
.metric-card.blue   { border-left: 3px solid #63b3ed; }
.metric-card.orange { border-left: 3px solid #ed8936; }

/* ── Section divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #2d3748;
    margin: 32px 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1f2937 !important;
}
[data-testid="stSidebar"] * { color: #d1d5db !important; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #1a202c;
    border: 2px dashed #2d3748;
    border-radius: 12px;
    padding: 8px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #4a90d9; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1a202c !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
}

/* ── SHAP word chips ── */
.chip-pos {
    display: inline-block;
    background: rgba(72,187,120,0.15);
    color: #68d391;
    border: 1px solid rgba(72,187,120,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 13px;
    margin: 3px 4px;
    font-weight: 500;
}
.chip-neg {
    display: inline-block;
    background: rgba(252,129,129,0.15);
    color: #fc8181;
    border: 1px solid rgba(252,129,129,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 13px;
    margin: 3px 4px;
    font-weight: 500;
}
.review-card {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.review-text { color: #a0aec0; font-size: 13.5px; line-height: 1.6; margin-bottom: 10px; }
.review-label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Topic keyword pills ── */
.keyword-pill {
    display: inline-block;
    background: rgba(99,179,237,0.1);
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 3px 11px;
    font-size: 12px;
    margin: 2px 3px;
}
.topic-card {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.topic-num {
    color: #4a90d9;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #2b6cb0, #2c5282) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    width: 100%;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 14px rgba(43,108,176,0.35) !important;
}
.stDownloadButton > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme helper ──────────────────────────────────────────────────
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

COLORS = {
    "Positive": "#48bb78",
    "Neutral":  "#ecc94b",
    "Negative": "#fc8181",
}

def apply_layout(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-size:22px; font-weight:700; color:#e2e8f0; margin-bottom:4px;">🔍 Review Intel</div>
        <div style="font-size:12px; color:#4a5568;">BUS5003 · Group 10 · La Trobe</div>
    </div>
    """, unsafe_allow_html=True)

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
    <h1>Amazon AirTag<br>Review Intelligence</h1>
    <p>Transform unstructured customer reviews into actionable business intelligence
    using Sentiment Analysis, Topic Modelling, and SHAP Explainability.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – DATA UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
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

df_raw = pd.read_csv(uploaded_file)

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

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-header">
    <div class="step-badge">2</div>
    <div>
        <p class="step-title">Text Preprocessing</p>
        <p class="step-sub">PII removal · lowercasing · stopword filtering · lemmatization</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Cleaning and tokenising reviews…"):
    df = preprocess_dataframe(df_raw, text_col)

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
    st.dataframe(
        df[[text_col, 'cleaned_text', 'token_str']].head(5),
        use_container_width=True,
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
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
    df = analyze_dataframe(df, text_col='cleaned_text')

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
    fig_pie = px.pie(
        df, names='label',
        title="Sentiment Distribution",
        color='label',
        color_discrete_map=COLORS,
        hole=0.50,
    )
    fig_pie.update_traces(textposition='outside', textinfo='percent+label',
                          textfont_color='#a0aec0')
    apply_layout(fig_pie)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    fig_hist = px.histogram(
        df, x='compound', nbins=40,
        title="Compound Score Distribution",
        color_discrete_sequence=['#4a90d9'],
        labels={'compound': 'Compound Score'},
    )
    fig_hist.add_vline(x= 0.05, line_dash='dash', line_color='#48bb78',
                       annotation_text='Positive', annotation_font_color='#48bb78')
    fig_hist.add_vline(x=-0.05, line_dash='dash', line_color='#fc8181',
                       annotation_text='Negative', annotation_font_color='#fc8181')
    apply_layout(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)

if rating_col:
    fig_bar = px.bar(
        df.groupby([rating_col, 'label']).size().reset_index(name='count'),
        x=rating_col, y='count', color='label', barmode='group',
        title="Sentiment by Star Rating",
        color_discrete_map=COLORS,
        labels={rating_col: 'Star Rating', 'count': 'Reviews'},
    )
    apply_layout(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True)

if risk_count:
    st.markdown("""
    <div style="color:#ed8936; font-weight:600; font-size:14px; margin: 16px 0 8px 0;">
        ⚠️ High-Risk Reviews — Require Human Review
    </div>
    """, unsafe_allow_html=True)
    high_risk_df = (
        df[df['high_risk']][[text_col, 'label', 'compound', 'confidence']]
        .head(10)
    )
    st.dataframe(high_risk_df, use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – TOPIC MODELLING
# ══════════════════════════════════════════════════════════════════════════════
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
    token_lists  = df['tokens'].tolist()
    lda_model, vectorizer, doc_term_matrix, coherence = train_lda(token_lists, n_topics)
    topics       = get_topic_keywords(lda_model, vectorizer)
    df['dominant_topic'] = assign_dominant_topic(lda_model, doc_term_matrix)

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
        st.markdown(f"""
        <div class="topic-card">
            <div class="topic-num">{topic_name}</div>
            <div>{pills}</div>
        </div>
        """, unsafe_allow_html=True)

with col_t2:
    topic_counts = df['dominant_topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Reviews']
    topic_counts['Topic'] = 'Topic ' + topic_counts['Topic'].astype(str)
    topic_counts = topic_counts.sort_values('Topic')

    fig_topics = px.bar(
        topic_counts, x='Topic', y='Reviews',
        title="Reviews per Dominant Topic",
        color='Reviews',
        color_continuous_scale=[[0, '#2c5282'], [1, '#63b3ed']],
        text='Reviews',
    )
    fig_topics.update_traces(textposition='outside', textfont_color='#a0aec0')
    apply_layout(fig_topics)
    fig_topics.update_coloraxes(showscale=False)
    st.plotly_chart(fig_topics, use_container_width=True)

cross = df.groupby(['dominant_topic', 'label']).size().reset_index(name='count')
cross['dominant_topic'] = 'Topic ' + cross['dominant_topic'].astype(str)
fig_cross = px.bar(
    cross, x='dominant_topic', y='count', color='label',
    barmode='stack',
    title="Sentiment Distribution per Topic",
    color_discrete_map=COLORS,
    labels={'dominant_topic': 'Topic', 'count': 'Reviews'},
)
apply_layout(fig_cross)
st.plotly_chart(fig_cross, use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-header">
    <div class="step-badge">5</div>
    <div>
        <p class="step-title">SHAP Explainability</p>
        <p class="step-sub">Word-level contribution scores for high-risk reviews</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Training classifier & computing SHAP values…"):
    pipeline, _ = train_classifier(df)
    sample_texts = df[df['high_risk']]['cleaned_text'].head(10).tolist()

if sample_texts:
    explanations = get_shap_explanations(pipeline, sample_texts, top_n=3)
    st.markdown(f"""
    <div style="color:#a0aec0; font-size:13.5px; margin-bottom:16px;">
        Showing <b style="color:#e2e8f0;">{len(explanations)}</b> high-risk review(s).
        <span style="color:#48bb78;">●</span> Green = positive driver &nbsp;
        <span style="color:#fc8181;">●</span> Red = negative driver
    </div>
    """, unsafe_allow_html=True)

    for exp in explanations:
        label_color = "#48bb78" if exp['predicted_label'] == 'Positive' else (
                      "#fc8181" if exp['predicted_label'] == 'Negative' else "#ecc94b")
        chips = "".join(
            f'<span class="chip-pos">+{word} ({score:+.3f})</span>'
            if score > 0 else
            f'<span class="chip-neg">−{word} ({score:+.3f})</span>'
            for word, score in exp['top_words']
        )
        st.markdown(f"""
        <div class="review-card">
            <div class="review-text">"{exp['text'][:160]}{'…' if len(exp['text']) > 160 else ''}"</div>
            <div style="margin-bottom:8px;">
                <span class="review-label" style="color:{label_color};">
                    ● {exp['predicted_label']}
                </span>
            </div>
            <div>{chips}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No high-risk reviews in the current selection. Adjust the star filter or upload more data.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 – DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-header">
    <div class="step-badge">6</div>
    <div>
        <p class="step-title">Export Results</p>
        <p class="step-sub">Download the full enriched dataset as CSV</p>
    </div>
</div>
""", unsafe_allow_html=True)

output_cols = [text_col, 'cleaned_text', 'label', 'compound',
               'confidence', 'high_risk', 'dominant_topic']
output_cols = [c for c in output_cols if c in df.columns]
csv_bytes   = df[output_cols].to_csv(index=False).encode('utf-8')

col_dl, col_info = st.columns([1, 2])
with col_dl:
    st.download_button(
        label    = "⬇️ Download Full Analysis (CSV)",
        data     = csv_bytes,
        file_name= "nlp_analysis_results.csv",
        mime     = "text/csv",
    )
with col_info:
    st.markdown(f"""
    <div style="background:#1a202c; border:1px solid #2d3748; border-radius:10px;
                padding:16px 20px; color:#718096; font-size:13px; line-height:1.8;">
        <b style="color:#e2e8f0;">Exported columns:</b><br>
        {" · ".join(f"<code style='color:#63b3ed;'>{c}</code>" for c in output_cols)}
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#2d3748; font-size:12px; margin-top:48px; padding:24px 0;">
    BUS5003 · Group 10 · NLP Analytics Platform · La Trobe University 2026
</div>
""", unsafe_allow_html=True)
