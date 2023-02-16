from sklearn.cluster import KMeans
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def get_themes(num_themes, sentence_vectors):
    """
    given sentence vectors, cluster the vectors into themes
    Args:
        num_themes: Number of themes
        sentence_vectors: sentence vectors

    Returns: list of int "theme" for each sentence vector

    """
    km = KMeans(n_clusters=num_themes, n_init=10, random_state=0)
    df = pd.DataFrame(sentence_vectors)

    return km.fit_predict(df)


def get_vectors(sentences):
    """
    given a list of sentences, returns vectors using TFiDF vectorizer
    Args:
        sentences: list of sentences to vectorize

    Returns: matrix of sentence vectors

    """
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(sentences).toarray()
