"""
Topic Modelling using scikit-learn LDA + TF-IDF.
No C++ compilation required.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def train_lda(token_lists: list, n_topics: int = 5) -> tuple:
    """
    Train an LDA topic model using scikit-learn.

    Parameters
    ----------
    token_lists : list of lists of str  (already tokenised)
    n_topics    : int

    Returns
    -------
    lda_model       : fitted LatentDirichletAllocation
    vectorizer      : fitted CountVectorizer
    doc_term_matrix : sparse matrix
    coherence_score : float  (approximate, based on topic diversity)
    """
    # Join tokens back to strings for CountVectorizer
    texts = [' '.join(tokens) for tokens in token_lists]

    vectorizer = CountVectorizer(
        max_features = 1000,
        min_df       = 2,
        max_df       = 0.90,
    )
    doc_term_matrix = vectorizer.fit_transform(texts)

    lda_model = LatentDirichletAllocation(
        n_components  = n_topics,
        random_state  = 42,
        max_iter      = 15,
        learning_method = 'batch',
    )
    lda_model.fit(doc_term_matrix)

    # Approximate coherence: average pairwise word overlap across topics
    feature_names = vectorizer.get_feature_names_out()
    top_words_per_topic = []
    for topic in lda_model.components_:
        top_idx = topic.argsort()[-10:]
        top_words_per_topic.append(set(feature_names[top_idx]))

    overlaps = []
    for i in range(len(top_words_per_topic)):
        for j in range(i + 1, len(top_words_per_topic)):
            overlap = len(top_words_per_topic[i] & top_words_per_topic[j])
            overlaps.append(overlap)
    avg_overlap   = np.mean(overlaps) if overlaps else 0
    coherence_score = round(1 - (avg_overlap / 10), 4)   # higher = more distinct

    return lda_model, vectorizer, doc_term_matrix, coherence_score


def get_topic_keywords(lda_model, vectorizer, n_words: int = 8) -> dict:
    """Return the top keywords for each topic."""
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for i, topic in enumerate(lda_model.components_):
        top_idx  = topic.argsort()[-n_words:][::-1]
        keywords = [feature_names[j] for j in top_idx]
        topics[f'Topic {i + 1}'] = keywords
    return topics


def assign_dominant_topic(lda_model, doc_term_matrix) -> list:
    """Return the dominant topic (1-based) for each document."""
    topic_distributions = lda_model.transform(doc_term_matrix)
    dominant_topics = topic_distributions.argmax(axis=1) + 1   # 1-indexed
    return dominant_topics.tolist()
