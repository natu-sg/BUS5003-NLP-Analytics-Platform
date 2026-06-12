import numpy as np
import pandas as pd
import shap
from sklearn.pipeline                 import Pipeline
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.linear_model             import LogisticRegression

LABEL_MAP = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


def train_classifier(df: pd.DataFrame) -> tuple:
    """
    Train a TF-IDF + Logistic Regression classifier on the sentiment labels.
    Returns the fitted pipeline and the filtered DataFrame used for training.
    Raises ValueError if fewer than 2 sentiment classes are present.
    """
    df = df[df['label'].isin(LABEL_MAP)].copy()
    df = df.dropna(subset=['cleaned_text'])
    df = df[df['cleaned_text'].str.strip() != '']
    df['label_num'] = df['label'].map(LABEL_MAP)

    n_classes = df['label_num'].nunique()
    if n_classes < 2:
        raise ValueError(
            f"Need at least 2 sentiment classes to train classifier, "
            f"but only found: {df['label'].unique().tolist()}"
        )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
        ('clf',   LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipeline.fit(df['cleaned_text'], df['label_num'])
    return pipeline, df


def get_shap_explanations(pipeline, texts: list, top_n: int = 3) -> list:
    """
    Compute SHAP values for a list of texts using a LinearExplainer.

    Returns a list of dicts:
        { 'text', 'predicted_label', 'top_words': [(word, shap_val), ...] }
    """
    vectorizer   = pipeline.named_steps['tfidf']
    clf          = pipeline.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()

    X        = vectorizer.transform(texts)
    explainer = shap.LinearExplainer(clf, X, feature_perturbation='interventional')
    shap_vals = explainer.shap_values(X)   # list of arrays, one per class

    results = []
    for i, text in enumerate(texts):
        pred_class = int(clf.predict(X[i])[0])
        sv         = shap_vals[pred_class][i]
        top_idx    = np.argsort(np.abs(sv))[-top_n:][::-1]
        top_words  = [(feature_names[j], round(float(sv[j]), 4)) for j in top_idx]

        results.append({
            'text':            text[:120] + '…' if len(text) > 120 else text,
            'predicted_label': LABEL_INV[pred_class],
            'top_words':       top_words,
        })
    return results
