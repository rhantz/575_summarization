from sklearn.cluster import KMeans
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from build_theme_graph import ThemeGraph


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


def majority_order(article_themes: list, selected_themes: list, article_lengths: list) -> list:
    """
    given a list of articles themes (sorted by article date, then sentence position),
    selected themes (any order), and the lengths of each article (sorted by article date),
    returns majority ordering of themes. Uses theme graph algortihm.

    Args:
        article_themes: list of article themes (list of ints)
        selected_themes: list of selected themes (list of ints)
        article_lengths: lengths of each article (must sum to length of article_themes)

    Returns: list of themes after majority ordering

    """
    if sum(article_lengths) != len(article_themes):
        raise ValueError(f"lengths of articles ({article_lengths}) must sum to length of article themes ({len(article_themes)})")

    themes_split_by_article = []
    for length in article_lengths:
        themes = article_themes[:length]
        article_themes = article_themes[length:]
        themes_split_by_article.append(themes)

    theme_weights = {theme: Counter() for theme in selected_themes}
    for article_themes in themes_split_by_article:
        for idx, theme in enumerate(article_themes):
            themes_after = set(article_themes[idx+1:])
            if theme in theme_weights:
                theme_weights[theme].update(themes_after)

    graph = ThemeGraph()
    for from_theme, weights in theme_weights.items():
        for to_theme, weight in weights.items():
            if to_theme in selected_themes:
                graph.add_arc_weight(from_theme, to_theme, weight)

    return graph.theme_order()

