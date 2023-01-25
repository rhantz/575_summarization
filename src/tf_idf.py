from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
from os.path import isfile, join, isdir
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import time
start_time = time.time()


# Function to read in an article and return it as a single string
def read_article(file_path):
    with open(file_path, 'r') as file:
        article_string = file.read()
    return article_string

#take in a sent that is tokenized for each word
def sent_score(doc_num, sent, tfidf_dic ):
    score = 0
    for word in sent:
        if word in tfidf_dic:
            score += tfidf_dic[word][doc_num]
    return score


directory = "/Users/tashi/Desktop/Ling575/575_summarization/outputs/devtest"
topic_ids = sorted([d for d in listdir(directory) if not isfile(join(directory, d))])
all_doc = []
for topic_id in sorted(topic_ids):
    topic_directory = f"/Users/tashi/Desktop/Ling575/575_summarization/outputs/devtest/{topic_id}"
    articles = sorted([f for f in listdir(topic_directory) if isfile(join(topic_directory, f))])
    for name in articles:
        article_string = read_article(f"/Users/tashi/Desktop/Ling575/575_summarization/outputs/devtest/{topic_id}/{name}")

        all_doc.append(article_string)


stop_word = stopwords.words('english')
print("########################################")
# Create the TfidfVectorizer object
vectorizer = TfidfVectorizer(stop_words = stop_word, token_pattern=r'\b[^\s]+\b')
# Fit and transform the documents to a matrix of TF-IDF features
tfidf_matrix = vectorizer.fit_transform(all_doc)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

tfidf_dict1 = dict(zip(feature_names, tfidf_matrix.toarray().T))

# print(all_doc)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

word_token = []
for iindex, doc in enumerate(all_doc):
    sent_text = sent_tokenize(doc)
    print(sent_text)
    for sentence in sent_text:
        para = sentence.split("\n\n")
        filtered_lines = [line for line in para if not line.startswith("headline:") and not line.startswith("date-time:")]
        filtered_text = "\n".join(filtered_lines)
        tokens = word_tokenize(filtered_text)
        print(tokens)
        score = sent_score(iindex, tokens,tfidf_dict1)
        print(filtered_text, ": ", score)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


end_time = time.time()
cpu_time = end_time - start_time
# cpu_time_mins = cpu_time / 60
print("CPU time in second:", cpu_time)
