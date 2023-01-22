from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from nltk.stem.snowball import SnowballStemmer
from nltk.util import bigrams
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from pulp import GLPK
from itertools import zip_longest
from collections import Counter



def remove_rare_concepts(concepts):
    # TODO
    """Removes concepts that occur in less than 3 documents"""
    return Counter({concept: weight for concept, weight in concepts.items() if weight >= 3})


def build_occurence_matrix(concepts, sentences):
    # TODO
    """builds tuple-indexed dictionary (O_ij) of whether concept_i appears in sentence_j

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

def get_sentence_concepts(sent):
    sent_stemmed = [stemmer.stem(word) for word in tokenizer.tokenize(sent)]
    sent_concepts = {bigram for bigram in bigrams(sent_stemmed) if bigram[0] not in stop_words and bigram[1] not in stop_words}
    return sent_concepts

def process_article(article_text):
    paragraphs = article_text.strip().split("\n\n")

    sents = []
    concepts = set()
    for paragraph in paragraphs:
        if ("headline:" not in paragraph) and ("date-time:" not in paragraph):
            sentences = paragraph.split("\n")
            for sent in sentences:
                sent_concepts = get_sentence_concepts(sent)
                concepts = concepts.union(sent_concepts)
                sents.append({"text": sent, "concepts": sent_concepts})

    return sents, concepts


def read_sentences(topic_id):

    topic_directory = f"../outputs/devtest/{topic_id}"
    articles = [f for f in listdir(topic_directory) if isfile(join(topic_directory, f))]

    # sorts articles by date
    articles = sorted(articles, key=lambda x: float(x[3:]))

    all_sents = []
    concept_weights = Counter()
    for idx, article in enumerate(articles):
        article = open(join(topic_directory, article), "r").read()
        sents, article_concepts = process_article(article)
        all_sents.append(sents)
        concept_weights.update(article_concepts)

    # orders sentences by place in article then by article date
    all_sents = [sent for sentences in zip_longest(*all_sents) for sent in sentences if sent is not None]

    return all_sents, concept_weights


if __name__ == '__main__':

    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words("english"))
    max_length = 100

    sentences, concept_weights = read_sentences("D1001A-A")
    concept_weights = remove_rare_concepts(concept_weights)
    concepts = sorted(concept_weights.keys())

    # Build occurence matrix
    occurence = build_occurence_matrix(concepts, sentences)

    # Implement ILP model

    model = LpProblem(name="content-selector", sense=LpMaximize)

    # Define the decision variables

    s = {j: LpVariable(name=f"s{j}", cat="Binary") for j in range(0, len(sentences))}
    c = {i: LpVariable(name=f"c{i}", cat="Binary") for i in range(0, len(concepts))}

    # Add constraints

    length_constraint = []
    for j, sentence in enumerate(sentences):
        length_constraint.append(len(sentence["text"].strip().split()) * s[j])
    model += (lpSum(length_constraint) <= max_length, "length_constraint")

    for i, concept in enumerate(concepts):
        coverage_constraint_1 = []

        for j, sentence in enumerate(sentences):
            coverage_constraint_1.append(s[j] * occurence[(i, j)])
            model += (s[j] * occurence[(i, j)] <= c[i], f"c{i}_in_s{j}_constraint")

        model += (lpSum(coverage_constraint_1) >= c[i], f"c{i}_in_at_least_one_s_constraint")

    # Set the objective function
    objective_function = []
    for i, concept in enumerate(concepts):
        objective_function.append(concept_weights[concept] * c[i])
    model += lpSum(objective_function)

    status = model.solve(solver=GLPK(msg=False))

    sentences_in_summary = []
    for var in s.values():
        if var.value() == 1:
            sentences_in_summary.append(int(var.name[1:]))

    for idx in sorted(sentences_in_summary):
        print(sentences[idx]["text"])












    # print("")
