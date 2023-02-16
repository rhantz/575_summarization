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
from itertools import zip_longest, chain
from collections import Counter
import export_summary
import argparse
from cluster import get_themes, get_vectors, majority_order
from collections import defaultdict
import spacy

nltk.download('stopwords')


def build_theme_matrix(themes: list) -> defaultdict:
    """
    given a list of themes that correspond to sentences by index, builds theme_matrix
    Args:
        themes: list of themes (ints)

    Returns: theme_matrix {(sentence_j, theme): occurrence} where occurrence is [0, 1]

    """
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


def build_concept_matrix(concepts: list, sentences: dict) -> dict:
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
    if args.concept_type == "named_entity":
        doc = nlp(sent)
        sent_concepts = set([entity.text.lower() for entity in doc.ents])
        return sent_concepts

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


def read_sentences(topic_id: str) -> tuple:
    """
    Reads each sentence available for a single topic id
    Args:
        topic_id: topic id directing function to read from correct topic_directory

    Returns:
        all_sents: list of sentence dictionaries for each article
        concepts: Counter of each concept and their weight (number of articles present in)
        article_lengths: stores length (number of sentences) of each article for downstream ordering purposes

    """

    topic_directory = f"../outputs/devtest/{topic_id}"
    articles = [f for f in listdir(topic_directory) if isfile(join(topic_directory, f))]

    # sorts articles by date (the date is the last 13 characters of the article title)
    articles = sorted(articles, key=lambda x: float(x[-13:]))

    all_sents = []
    lengths = []
    concept_weights = Counter()
    for idx, article in enumerate(articles):
        article = open(join(topic_directory, article), "r").read()
        # Retrieves sentences dictionary and concepts for article
        sents, article_concepts = process_article(article)
        all_sents.append(sents)
        lengths.append(len(sents))
        # update the weights of each concept by adding in current articles concepts to Counter
        concept_weights.update(article_concepts)

    if not args.majority_order:
        # default Information Ordering - orders sentences sentence position, then article date
        # e.g [sentence_1_article_1, sentence_1_article_2, ... , last_sentence_last_article]
        all_sents = [sent for sentences in zip_longest(*all_sents) for sent in sentences if sent is not None]
    else:
        # keeps ordering by article date
        all_sents = list(chain.from_iterable(all_sents))

    return all_sents, concept_weights, lengths


if __name__ == '__main__':

    ###
    # Collects all arguments
    ###

    # stems all but stop words
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    # tokenizes and removes punctuation (does not remove _ )
    tokenizer = RegexpTokenizer(r'\w+')
    # set of English stop words
    stop_words = set(stopwords.words("english"))
    # max length (number of whitespace delimited tokens) in each summary must be 100
    max_length = 100

    # parses CLI arguments
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
        "--num_themes", type=int, required=False, default=0, help="number of themes to cluster sentences into (if unspecified or 0, theme constraint not applied)")
    parser.add_argument(
        "--theme_redundancy", type=int, required=False, default=1, help="max times each theme can be selected from (if unspecified, defaults to 1)")
    parser.add_argument(
        "--majority_order", action="store_true", help="option to sort sentences by majority_ordering (if unspecified, sorts sentences by sentence position, then article date)")

    args = parser.parse_args(sys.argv[1:])

    if args.num_themes == 0 and args.majority_order:
        raise ValueError("to sort by majority order, you must specify a number of themes with the --num_themes parameter")

    if args.concept_type == "named_entity":
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

    # iterates through each topic_id set of articles
    directory = f"../outputs/{args.input_dir}/"
    topic_ids = [d for d in listdir(directory) if not isfile(join(directory, d))]

    for topic_id in topic_ids:

        # Retrieves sentences, concepts, and concept_weights
        sentences, concept_weights, article_lengths = read_sentences(topic_id)
        concept_weights = remove_rare_concepts(concept_weights)
        concepts = sorted(concept_weights.keys())

        # Builds concept_occurence matrix
        concept_occurence = build_concept_matrix(concepts, sentences)

        if args.num_themes != 0:
            # Retrieves sentence_vectors and themes
            sentence_vectors = get_vectors([sentence["text"] for sentence in sentences])
            themes = get_themes(num_themes=args.num_themes, sentence_vectors=sentence_vectors) + 1

        # Builds theme_occurence matrix
        if args.num_themes != 0:
            theme_occurence = build_theme_matrix(themes)

        ###
        # Implements ILP model
        ###

        model = LpProblem(name="content-selector", sense=LpMaximize)

        # Defines the decision variables
        s = {j: LpVariable(name=f"s{j}", cat="Binary") for j in range(0, len(sentences))}
        c = {i: LpVariable(name=f"c{i}", cat="Binary") for i in range(0, len(concepts))}

        # Adds constraints and objective function to optimize

        length_constraint = []
        for j, sentence in enumerate(sentences):

            # Retrieves length of each sentence
            l = len(sentence["text"].strip().split())

            if args.min_sent_length != 0:

                # s_j Length Constraint: s_j is above the minimum sentence length
                model += (l * s[j] >= args.min_sent_length * s[j], f"s{j}_length_constraint")

            length_constraint.append(l * s[j])

        # Length Constraint: All sentences chosen do not exceed 100 tokens
        model += (lpSum(length_constraint) <= max_length, "overall_length_constraint")

        for i, concept in enumerate(concepts):
            coverage_constraint_1 = []

            for j, sentence in enumerate(sentences):
                coverage_constraint_1.append(s[j] * concept_occurence[(i, j)])

                # Coverage Constraint 2: If s_j is included, so is c_i, if c_i in s_j
                model += (s[j] * concept_occurence[(i, j)] <= c[i], f"c{i}_in_s{j}_constraint")

            # Coverage Constraint 1: If c_i is included, at least 1 sentence that has c_i is included
            model += (lpSum(coverage_constraint_1) >= c[i], f"c{i}_in_at_least_one_s_constraint")

        if args.num_themes != 0:
            for theme in set(themes):
                theme_constraint = []
                for j, sentence in enumerate(sentences):
                    theme_constraint.append(s[j] * theme_occurence[(j, theme)])

                # Theme Constraint: theme_t can only occur at most {args.theme_redundancy} time(s) in the selected sentences
                model += (lpSum(theme_constraint) <= args.theme_redundancy, f"theme_{theme}_constraint")

        objective_function = []
        for i, concept in enumerate(concepts):
            objective_function.append(concept_weights[concept] * c[i])

        # Objective Function: Maximizes the weighted sum of each included c_i and their respective weight
        model += lpSum(objective_function)

        # Solves the model -- decides which s_j's should be included
        status = model.solve(solver=GLPK(msg=False))

        # Collects the index (j) of each s_j that should be included (i.e. s_j = 1) in optimal solution
        sentences_in_summary = []
        # Collects the theme of each s_j that should be included in optimal solution
        themes_in_summary = []
        for var in s.values():
            if var.value() == 1:
                sentences_in_summary.append(sentences[int(var.name[1:])]["text"])
                if args.num_themes != 0:
                    themes_in_summary.append(themes[int(var.name[1:])])

        ###
        # Information Ordering
        ###

        # performs majority ordering on sentences
        if args.majority_order:

            # orders themes
            order_of_themes = majority_order(themes, themes_in_summary, article_lengths)

            # orders sentences by order_of_themes
            theme_to_sents = {}
            for theme, sentence in zip(themes_in_summary, sentences_in_summary):
                if theme in theme_to_sents:
                    theme_to_sents[theme].append(sentence)
                else:
                    theme_to_sents[theme] = [sentence]
            sentences_in_summary = list(chain.from_iterable([theme_to_sents[theme] for theme in order_of_themes]))

        # Prints selected sentences to file
        export_summary.export_summary(sentences_in_summary, topic_id[:6], "2", "../outputs/D3")