import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text: str) -> dict:
    """
    Run VADER on a single text.
    Returns compound score, pos/neg/neu scores, label, confidence,
    and a high_risk flag for low-confidence predictions.
    """
    scores   = _analyzer.polarity_scores(str(text))
    compound = scores['compound']

    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'

    confidence = round(abs(compound), 4)   # 0–1, higher = more confident
    high_risk  = confidence < 0.20          # flag ambiguous reviews

    return {
        'compound':   round(compound,      4),
        'pos_score':  round(scores['pos'], 4),
        'neg_score':  round(scores['neg'], 4),
        'neu_score':  round(scores['neu'], 4),
        'label':      label,
        'confidence': confidence,
        'high_risk':  high_risk,
    }


def analyze_dataframe(df: pd.DataFrame, text_col: str = 'cleaned_text') -> pd.DataFrame:
    """Apply VADER sentiment analysis to every row in a DataFrame."""
    df      = df.copy()
    results = df[text_col].apply(get_sentiment)
    return pd.concat([df.reset_index(drop=True),
                      pd.DataFrame(list(results))], axis=1)
