import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt',       quiet=True)
nltk.download('punkt_tab',   quiet=True)
nltk.download('stopwords',   quiet=True)
nltk.download('wordnet',     quiet=True)


def remove_pii(text: str) -> str:
    """Mask emails and phone numbers (DR04 compliance)."""
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\b\d{7,}\b', '[PHONE]', text)
    return text


def clean_text(text: str) -> str:
    """Lowercase, remove PII, strip special characters."""
    text = str(text).lower()
    text = remove_pii(text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list:
    """
    Tokenise, remove stopwords (keeping sentiment-critical words),
    and lemmatize each token.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words  = set(stopwords.words('english'))

    # Keep words critical to sentiment
    sentiment_critical = {'not', 'no', 'never', 'very', 'too', 'but', 'however'}
    stop_words -= sentiment_critical

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in stop_words and len(t) > 2
    ]
    return tokens


def preprocess_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Full preprocessing pipeline applied to an entire DataFrame."""
    df = df.copy()
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df['tokens']       = df['cleaned_text'].apply(tokenize_and_lemmatize)
    df['token_str']    = df['tokens'].apply(lambda x: ' '.join(x))
    return df
