"""
BUS5003 – NLP Analytics Platform
Amazon AirTag Review Intelligence
Group 10 | La Trobe University
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocess      import preprocess_dataframe
from utils.sentiment       import analyze_dataframe
from utils.topics          import train_lda, get_topic_keywords, assign_dominant_topic

from utils.explainability  import train_classifier, get_shap_explanations

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "NLP Analytics Platform",
    page_icon  = "🔍",
    layout     = "wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 NLP Analytics Platform")
    st.markdown("**Amazon AirTag Review Intelligence**")
    st.markdown("*Group 10 – BUS5003*")
    st.divider()

    st.subheader("⚙️ Settings")
    n_topics = st.slider("Number of LDA Topics", min_value=3, max_value=10, value=5)

    star_filter = st.multiselect(
        "Filter by Star Rating",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
    )

    st.divider()
    st.caption("Sprint 1 & 2 – Release 1")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 NLP Analytics Platform")
st.markdown(
    "**Amazon AirTag Review Intelligence** — "
    "Transforming unstructured customer reviews into actionable business intelligence "
    "via Sentiment Analysis & Topic Modelling."
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – DATA UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.header("📂 Step 1: Upload Review Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing customer reviews",
    type=["csv"],
    help="CSV must have at least one text column. Optional: a 'rating' or 'stars' column."
)

if not uploaded_file:
    st.info("👆 Upload a CSV to begin. No file yet? Use the sample data in `data/sample_reviews.csv`.")
    st.stop()

# ── Load & validate ───────────────────────────────────────────────────────────
df_raw = pd.read_csv(uploaded_file)
st.success(f"✅ Loaded **{len(df_raw):,}** rows, **{len(df_raw.columns)}** columns.")

text_cols = df_raw.select_dtypes(include='object').columns.tolist()
if not text_cols:
    st.error("No text columns found in the uploaded file.")
    st.stop()

col_left, col_right = st.columns(2)
with col_left:
    text_col = st.selectbox("Select the review text column", text_cols)

# Detect rating column
rating_col = None
for candidate in ['rating', 'stars', 'score', 'Rating', 'Stars']:
    if candidate in df_raw.columns:
        rating_col = candidate
        break

with col_right:
    if rating_col:
        df_raw[rating_col] = pd.to_numeric(df_raw[rating_col], errors='coerce')
        df_raw = df_raw[df_raw[rating_col].isin(star_filter)]
        st.info(f"Filtered to **{len(df_raw):,}** reviews (stars: {star_filter})")
    else:
        st.caption("No rating column detected — showing all reviews.")

with st.expander("Preview raw data"):
    st.dataframe(df_raw.head(10), use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
st.header("⚙️ Step 2: Text Preprocessing")

with st.spinner("Cleaning text, removing PII, tokenising, lemmatizing…"):
    df = preprocess_dataframe(df_raw, text_col)

c1, c2, c3 = st.columns(3)
c1.metric("Total Reviews",       f"{len(df):,}")
c2.metric("Avg Tokens / Review", f"{df['tokens'].apply(len).mean():.1f}")
c3.metric("PII Masking",         "✅ Applied")

with st.expander("Show preprocessed samples"):
    st.dataframe(
        df[[text_col, 'cleaned_text', 'token_str']].head(5),
        use_container_width=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.header("💬 Step 3: Sentiment Analysis (VADER)")

with st.spinner("Running VADER sentiment analysis…"):
    df = analyze_dataframe(df, text_col='cleaned_text')

counts = df['label'].value_counts()

m1, m2, m3, m4 = st.columns(4)
m1.metric("🟢 Positive",       counts.get('Positive', 0))
m2.metric("🟡 Neutral",        counts.get('Neutral',  0))
m3.metric("🔴 Negative",       counts.get('Negative', 0))
m4.metric("⚠️ High-Risk",      int(df['high_risk'].sum()),
          help="Reviews with confidence < 0.20 — needs human review")

col_a, col_b = st.columns(2)

with col_a:
    fig_pie = px.pie(
        df, names='label',
        title="Sentiment Distribution",
        color='label',
        color_discrete_map={
            'Positive': '#2ecc71',
            'Neutral':  '#f39c12',
            'Negative': '#e74c3c',
        },
        hole=0.35,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    fig_hist = px.histogram(
        df, x='compound', nbins=40,
        title="Compound Score Distribution",
        color_discrete_sequence=['#3498db'],
        labels={'compound': 'Compound Score'},
    )
    fig_hist.add_vline(x= 0.05, line_dash='dash', line_color='green',
                       annotation_text='Positive threshold')
    fig_hist.add_vline(x=-0.05, line_dash='dash', line_color='red',
                       annotation_text='Negative threshold')
    st.plotly_chart(fig_hist, use_container_width=True)

if rating_col:
    fig_bar = px.bar(
        df.groupby([rating_col, 'label']).size().reset_index(name='count'),
        x=rating_col, y='count', color='label', barmode='group',
        title="Sentiment by Star Rating",
        color_discrete_map={
            'Positive': '#2ecc71',
            'Neutral':  '#f39c12',
            'Negative': '#e74c3c',
        },
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("⚠️ High-Risk Reviews (Confidence < 0.20)")
st.caption("These reviews have ambiguous sentiment and require human review.")
high_risk_df = (
    df[df['high_risk']][[text_col, 'label', 'compound', 'confidence']]
    .head(10)
)
st.dataframe(high_risk_df, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – TOPIC MODELLING
# ══════════════════════════════════════════════════════════════════════════════
st.header("🗂️ Step 4: Topic Modelling (LDA + TF-IDF)")

with st.spinner(f"Training LDA with {n_topics} topics — this may take ~30 seconds…"):
    token_lists = df['tokens'].tolist()
    lda_model, vectorizer, doc_term_matrix, coherence = train_lda(token_lists, n_topics)
    topics      = get_topic_keywords(lda_model, vectorizer)
    df['dominant_topic'] = assign_dominant_topic(lda_model, doc_term_matrix)

st.metric(
    "C_v Coherence Score", f"{coherence:.4f}",
    help="0.50–0.70 is acceptable. Higher = more coherent topics."
)

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.subheader("Top Keywords per Topic")
    for topic_name, words in topics.items():
        st.markdown(f"**{topic_name}:** {', '.join(words)}")

with col_t2:
    topic_counts = df['dominant_topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Review Count']
    fig_topics = px.bar(
        topic_counts, x='Topic', y='Review Count',
        title="Reviews per Dominant Topic",
        color='Review Count',
        color_continuous_scale='Blues',
    )
    st.plotly_chart(fig_topics, use_container_width=True)

cross = df.groupby(['dominant_topic', 'label']).size().reset_index(name='count')
fig_cross = px.bar(
    cross, x='dominant_topic', y='count', color='label',
    barmode='stack',
    title="Sentiment Distribution per Topic",
    color_discrete_map={
        'Positive': '#2ecc71',
        'Neutral':  '#f39c12',
        'Negative': '#e74c3c',
    },
    labels={'dominant_topic': 'Topic', 'count': 'Reviews'},
)
st.plotly_chart(fig_cross, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔬 Step 5: SHAP Explainability")
st.markdown(
    "Each high-risk review is explained by its **top-3 contributing words** "
    "(SHAP values). 🟢 = pushes toward positive, 🔴 = pushes toward negative."
)

with st.spinner("Training classifier & computing SHAP values…"):
    pipeline, _ = train_classifier(df)
    sample_texts = df[df['high_risk']]['cleaned_text'].head(10).tolist()

if sample_texts:
    explanations = get_shap_explanations(pipeline, sample_texts, top_n=3)
    for exp in explanations:
        with st.expander(f"📝 {exp['text'][:90]}… → **{exp['predicted_label']}**"):
            for word, score in exp['top_words']:
                icon = '🟢' if score > 0 else '🔴'
                st.write(f"{icon} **{word}** — SHAP value: `{score:+.4f}`")
else:
    st.info("No high-risk reviews in the current selection. Adjust the star filter or upload more data.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 – DOWNLOAD RESULTS
# ══════════════════════════════════════════════════════════════════════════════
st.header("📥 Step 6: Download Results")

output_cols = [text_col, 'cleaned_text', 'label', 'compound',
               'confidence', 'high_risk', 'dominant_topic']
output_cols = [c for c in output_cols if c in df.columns]

csv_bytes = df[output_cols].to_csv(index=False).encode('utf-8')

st.download_button(
    label    = "⬇️ Download Full Analysis (CSV)",
    data     = csv_bytes,
    file_name= "nlp_analysis_results.csv",
    mime     = "text/csv",
)

st.caption("BUS5003 Group 10 | NLP Analytics Platform | La Trobe University 2026")
