from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.util import bigrams, skipgrams
from nltk.corpus import stopwords
import sys
from os import listdir
from os.path import isfile, join
from pulp import GLPK
from itertools import zip_longest
from collections import Counter
import export_summary
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from cluster import get_themes
from collections import defaultdict

nltk.download('stopwords')


# TODO - docstring
def build_theme_matrix(themes):
    theme_matrix = defaultdict(int)
    for t, theme in enumerate(themes):
        theme_matrix[(t, theme)] = 1
    return theme_matrix


def remove_rare_concepts(concepts: Counter) -> Counter:
    """
    Removes concepts that occur in less than 3 documents
    Args:
        concepts: Counter for occurence of concept in each document
         format - {concept: number of documents}

    Returns:
        Counter with only concepts that occur in 3 or more documents
    """
    return Counter({concept: weight for concept, weight in concepts.items() if weight >= 3})


def build_occurence_matrix(concepts: list, sentences: dict) -> dict:
    """
    builds tuple-indexed dictionary (O_ij) of whether concept_i appears in sentence_j
        {(i, j) : [0, 1]}

    Args:
        concepts: list of concepts
        sentences: dictionary holding sentences {"text": full sentence, "concepts": set of concepts in sentence}

    Returns:
        occurence: tuple-indexed dictionary (O_ij) of whether concept_i appears in sentence_j
            {(i, j) : [0, 1]}
    """
    occurence = {}
    for i, concept_i in enumerate(concepts):
        for j, sentence_j in enumerate(sentences):
            if concept_i in sentence_j["concepts"]:
                occurence[(i, j)] = 1
            else:
                occurence[(i, j)] = 0
    return occurence


def get_sentence_concepts(sent: str) -> set:
    #TODO - add NE?
    """
    For each sentence: stems and returns set of desired concepts

    Args:
        sent: string sentence

    Returns:
        sent_concepts: set of desired concepts

    """
    if args.remove_punctuation:
        sent_stemmed = [stemmer.stem(word) for word in tokenizer.tokenize(sent)]
    else:
        sent_stemmed = [stemmer.stem(word) for word in sent.strip().split(" ")]

    if args.remove_stop_words:
        sent_stemmed = [word for word in sent_stemmed if word not in stop_words]

    # stop words are always removed for unigrams
    if args.concept_type == "unigrams":
        sent_concepts = {unigram for unigram in sent_stemmed if unigram not in stop_words}

    # for bigrams and skipgrams, only 1 stop word is allowed (unless stop words are fully removed)
    elif args.concept_type == "bigrams":
        sent_concepts = {bigram for bigram in bigrams(sent_stemmed) if (bigram[0] not in stop_words) or (bigram[1] not in stop_words)}
    elif args.concept_type == "skipgrams":
        sent_concepts = {skipgram for skipgram in skipgrams(sent_stemmed, 2, args.skipgram_degree) if (skipgram[0] not in stop_words) or (skipgram[1] not in stop_words)}
    else:
        raise ValueError(
            f"{args.concept_type} is not a valid format of concept, please select 'unigrams', 'bigrams', or 'skipgrams'")
    return sent_concepts


def process_article(article_text: str) -> tuple:
    """
    Processes article to return sentences dictionary and set of all concepts in article
    Args:
        article_text: article text in string format

    Returns:
        sents: dictionary holding all sentences in article
            {"text": full sentence, "concepts": set of concepts in sentence}
        concepts: set of all concepts in article
    """
    # Splits article into paragraphs
    paragraphs = article_text.strip().split("\n\n")

    sents = []
    concepts = set()

    # Iterates through paragraphs
    for paragraph in paragraphs:
        # Only uses paragraphs that don't begin with "headline" or "date-time"
        if ("headline:" not in paragraph) and ("date-time:" not in paragraph):
            # Splits paragraph into sentences
            sentences = paragraph.split("\n")
            for sent in sentences:
                sent_concepts = get_sentence_concepts(sent)
                concepts = concepts.union(sent_concepts)
                sents.append({"text": sent, "concepts": sent_concepts})

    return sents, concepts


def read_sentences(topic_id: str, order_sents: bool = True) -> tuple:
    """
    Reads each sentence available for a single topic id
    Args:
        topic_id: topic id directing function to read from correct topic_directory
        order_sents: option to perform information ordering on sentences (True by default)

    Returns:
        all_sents: list of sentence dictionaries for each article
        concepts: Counter of each concept and their weight (number of articles present in)

    """

    topic_directory = f"../outputs/devtest/{topic_id}"
    articles = [f for f in listdir(topic_directory) if isfile(join(topic_directory, f))]

    # sorts articles by date (the date is the last 13 characters of the article title)
    articles = sorted(articles, key=lambda x: float(x[-13:]))

    all_sents = []
    concept_weights = Counter()
    for idx, article in enumerate(articles):
        article = open(join(topic_directory, article), "r").read()
        # Retrieves sentences dictionary and concepts for article
        sents, article_concepts = process_article(article)
        all_sents.append(sents)
        # update the weights of each concept by adding in current articles concepts to Counter
        concept_weights.update(article_concepts)

    # By default performs Information Ordering heuristic
    # if order_sents == False, sentences are left in order of article
    # TODO - add argument for this
    # TODO - figure out shape for if True
    if order_sents:
        # Information Ordering - orders sentences by position in article then by article date
        # e.g [sentence 1 article 1, sentence 1 article 2, ... , last sentence last article]
        all_sents = [sent for sentences in zip_longest(*all_sents) for sent in sentences if sent is not None]

    return all_sents, concept_weights


if __name__ == '__main__':

    # use TFiDF vectorizer to determine sentence theme
    vectorizer = TfidfVectorizer()
    # Stems all but stop words
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    # tokenizes and removes punctuation (does not remove _ )
    tokenizer = RegexpTokenizer(r'\w+')
    # set of stop words
    stop_words = set(stopwords.words("english"))
    # max length (number of whitespace delimited tokens) in each summary must be 100
    max_length = 100

    parser = argparse.ArgumentParser(
        description="Run ILP method of content selection for summarization system.")

    parser.add_argument(
        "--input_dir", type=str, required=True, help="input directory to preprocessed articles (training, devtest, evaltest)")
    parser.add_argument(
        "--concept_type", type=str, required=True, help="concept schema (unigrams, bigrams, skipgrams)")
    parser.add_argument(
        "--skipgram_degree", type=int, required=False, default=2, help="degree of skipgram (if unspecified, defaults to 2)")
    parser.add_argument(
        "--remove_punctuation", action="store_true", help="option to remove punctuation in concepts")
    parser.add_argument(
        "--remove_stop_words", action="store_true", help="option to remove stop words from concepts")
    parser.add_argument(
        "--min_sent_length", type=int, required=False, default=0, help="minimum length of selected sentences (if unspecified, defaults to 0)")
    parser.add_argument(
        "--num_themes", type=int, required=False, default=0, help="Number of themes to cluster sentences into")

    args = parser.parse_args(sys.argv[1:])

    directory = f"../outputs/{args.input_dir}/"
    topic_ids = [d for d in listdir(directory) if not isfile(join(directory, d))]

    for topic_id in topic_ids:

        sentences, concept_weights = read_sentences(topic_id)
        concept_weights = remove_rare_concepts(concept_weights)
        concepts = sorted(concept_weights.keys())

        # Builds occurence matrix
        occurence = build_occurence_matrix(concepts, sentences)

        sentences_text = [sentence["text"] for sentence in sentences]
        sentence_vectors = vectorizer.fit_transform(sentences_text).toarray()

        themes = get_themes(num_themes=5, sentence_vectors=sentence_vectors) + 1

        if args.num_themes != 0:
            theme_occurence = build_theme_matrix(themes)

        # Implement ILP model

        model = LpProblem(name="content-selector", sense=LpMaximize)

        # Define the decision variables

        s = {j: LpVariable(name=f"s{j}", cat="Binary") for j in range(0, len(sentences))}
        c = {i: LpVariable(name=f"c{i}", cat="Binary") for i in range(0, len(concepts))}

        # Add constraints

        # Length Constraint: All sentences chosen do not exceed 100 tokens
        length_constraint = []
        for j, sentence in enumerate(sentences):

            l = len(sentence["text"].strip().split())

            if args.min_sent_length != 0:
                # s_j Length Constraint: s_j is above the minimum sentence length
                model += (l * s[j] >= args.min_sent_length * s[j], f"s{j}_length_constraint")

            length_constraint.append(l * s[j])
        model += (lpSum(length_constraint) <= max_length, "overall_length_constraint")

        for i, concept in enumerate(concepts):
            # Coverage Constraint 1: If c_i is included, at least 1 sentence that has c_i is included
            coverage_constraint_1 = []

            for j, sentence in enumerate(sentences):
                coverage_constraint_1.append(s[j] * occurence[(i, j)])

                # Coverage Constraint 2: If s_j is included, so is c_i, if c_i in s_j
                model += (s[j] * occurence[(i, j)] <= c[i], f"c{i}_in_s{j}_constraint")

            model += (lpSum(coverage_constraint_1) >= c[i], f"c{i}_in_at_least_one_s_constraint")

        if args.num_themes != 0:
            for theme in set(themes):
                # Theme Constraint: theme_t can only occur at most once in the selected sentences
                # TODO - set a threshold for theme occurence?
                theme_constraint = []
                for j, sentence in enumerate(sentences):
                    theme_constraint.append(s[j] * theme_occurence[(j, theme)])
                model += (lpSum(theme_constraint) <= 1, f"theme_{theme}_constraint")

        # Objective Function: Maximize the weighted sum of each included c_i and their respective weight
        objective_function = []
        for i, concept in enumerate(concepts):
            objective_function.append(concept_weights[concept] * c[i])
        model += lpSum(objective_function)

        # Solve the model -- decide which s_j's should be included
        status = model.solve(solver=GLPK(msg=False))

        # Collect the index (j) of each s_j that should be included (i.e. s_j = 1 in optimal solution)
        sentences_in_summary = []
        themes_in_summary = []
        for var in s.values():
            if var.value() == 1:
                sentences_in_summary.append(sentences[int(var.name[1:])]["text"])
                themes_in_summary.append(themes[int(var.name[1:])])

        # Print to file
        export_summary.export_summary(sentences_in_summary, topic_id[:6], "2", "../outputs/D3")

        # TODO - rebase and import graph algo module :)

