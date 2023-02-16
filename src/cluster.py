from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def get_themes(num_themes, sentence_vectors):
    """
    given sentence vectors, cluster the vectors into themes
    Args:
        num_themes: Number of themes
        sentence_vectors: sentence vectors

    Returns:

    """
    km = KMeans(n_clusters=num_themes, n_init=10)
    df = pd.DataFrame(sentence_vectors)

    return km.fit_predict(df)

