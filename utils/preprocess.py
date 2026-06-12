import re
import nltk
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt',       quiet=True)
nltk.download('punkt_tab',   quiet=True)
nltk.download('stopwords',   quiet=True)
nltk.download('wordnet',     quiet=True)

# ── Build shared objects ONCE (not per-row) ───────────────────────────────────
_LEMMATIZER  = WordNetLemmatizer()
_STOP_WORDS  = set(stopwords.words('english')) - {'not', 'no', 'never', 'very', 'too', 'but', 'however'}

# Compiled regex patterns (faster than re.sub with string patterns)
_RE_EMAIL    = re.compile(r'\S+@\S+')
_RE_PHONE    = re.compile(r'\b\d{7,}\b')
_RE_NONALPHA = re.compile(r'[^a-zA-Z\s]')
_RE_SPACES   = re.compile(r'\s+')


# ── Per-row functions (use shared objects) ────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _RE_EMAIL.sub('[EMAIL]', text)
    text = _RE_PHONE.sub('[PHONE]', text)
    text = _RE_NONALPHA.sub(' ', text)
    text = _RE_SPACES.sub(' ', text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list:
    tokens = word_tokenize(text)
    return [
        _LEMMATIZER.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in _STOP_WORDS and len(t) > 2
    ]


# ── Vectorized + parallel pipeline ───────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame, text_col: str,
                         progress_callback=None) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    progress_callback(pct: float, msg: str) — called after each stage
    so the caller can update a Streamlit progress bar.
    """
    df = df.copy()
    total = len(df)

    # ── Stage 1: vectorized text cleaning (fast — no Python loop) ────────────
    if progress_callback:
        progress_callback(0.05, f"🧹 Cleaning text…  0 / {total:,} rows")

    series = df[text_col].astype(str).str.lower()
    series = series.str.replace(_RE_EMAIL,    '[EMAIL]',  regex=True)
    series = series.str.replace(_RE_PHONE,    '[PHONE]',  regex=True)
    series = series.str.replace(_RE_NONALPHA, ' ',        regex=True)
    series = series.str.replace(_RE_SPACES,   ' ',        regex=True).str.strip()
    df['cleaned_text'] = series

    if progress_callback:
        progress_callback(0.30, f"✅ Text cleaned  ({total:,} rows)")

    # ── Stage 2: parallel tokenization + lemmatization ───────────────────────
    texts      = df['cleaned_text'].tolist()
    BATCH_SIZE = 2_000
    n_batches  = max(1, (total + BATCH_SIZE - 1) // BATCH_SIZE)
    results    = [None] * total

    def process_batch(args):
        start, batch = args
        return start, [tokenize_and_lemmatize(t) for t in batch]

    batches = [
        (i * BATCH_SIZE, texts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        for i in range(n_batches)
    ]

    with ThreadPoolExecutor() as executor:
        for done, (start, tokens) in enumerate(
            executor.map(process_batch, batches), start=1
        ):
            for j, tok in enumerate(tokens):
                results[start + j] = tok
            if progress_callback:
                pct = 0.30 + 0.65 * (done / n_batches)
                rows_done = min((done) * BATCH_SIZE, total)
                progress_callback(
                    pct,
                    f"🔤 Tokenising & lemmatising…  {rows_done:,} / {total:,} rows"
                )

    df['tokens']    = results
    df['token_str'] = df['tokens'].apply(' '.join)

    if progress_callback:
        progress_callback(1.0, f"✅ Preprocessing complete — {total:,} rows")

    return df
