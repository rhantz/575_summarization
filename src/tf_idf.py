from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
from os.path import isfile, join, isdir
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import time
import pandas as pd
import numpy as np
start_time = time.time()


# Function to read in an article and return it as a single string
def read_article(file_path):
    with open(file_path, 'r') as file:
        article_string = file.read()
    return article_string

#take in a sent that is tokenized for each word
def sent_score(doc_num, sent, tfidf_dic ):
    score = 0
    word_num = 0
    for word in sent:
        if word in tfidf_dic:
            word_num += 1
            score += tfidf_dic[word][doc_num]

    if word_num > 0:
        return (score/word_num) 
    else:
        return 0
    

directory = "/Users/tashi/Desktop/Ling575/575_summarization/outputs/devtest"
topic_ids = sorted([d for d in listdir(directory) if not isfile(join(directory, d))])
total_top = len(topic_ids)
print("@@@@@@@@@@@@@@@")
print(topic_ids)
print("@@@@@@@@@@@@@@@")

all_doc = []
for topic_id in sorted(topic_ids):
    topic_directory = f"/Users/tashi/Desktop/Ling575/575_summarization/outputs/devtest/{topic_id}"
    articles = sorted([f for f in listdir(topic_directory) if isfile(join(topic_directory, f))])
    for name in articles:
        article_string = read_article(f"/Users/tashi/Desktop/Ling575/575_summarization/outputs/devtest/{topic_id}/{name}")

        all_doc.append(article_string)

stop_word = stopwords.words('english')
# Create the TfidfVectorizer object
vectorizer = TfidfVectorizer(stop_words = stop_word, token_pattern=r'\b[^\s]+\b')
# Fit and transform the documents to a matrix of TF-IDF features
tfidf_matrix = vectorizer.fit_transform(all_doc)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()
tfidf_dict1 = dict(zip(feature_names, tfidf_matrix.toarray().T))


sents = []
word_count = []
ave_score = []
article_num = []

for iindex, doc in enumerate(all_doc):
    sent_text = sent_tokenize(doc)
    for sentence in sent_text:
        para = sentence.split("\n\n")
        filtered_lines = [line for line in para if not line.startswith("headline:") and not line.startswith("date-time:")]
        filtered_text = "\n".join(filtered_lines)
        tokens = word_tokenize(filtered_text)
        score = sent_score(iindex, tokens,tfidf_dict1)
        sents.append(tokens)
        word_count.append(len(tokens))
        ave_score.append(score)
        article_num.append(iindex)

df = pd.DataFrame({'article_num':article_num, 'sentence': sents, 'word_count': word_count, 'tfidf_scores': ave_score})
df = df.sort_values(by = ['article_num', 'tfidf_scores'], ascending = [True, False])
df = df.reset_index()
df.drop(df.columns[0], axis=1, inplace=True)


total_art = iindex + 1 #460
flag = 0
word_limit = 100
minimum = 8

value_to_search_for = 0
sum = []
for j in range(0, total_art):
    my_tuple = (j, [])
    value_to_search_for = j 
    selected_rows = df[df.iloc[:,0] == value_to_search_for]
    w_c = 0
    for _, row in selected_rows.iterrows():
        # if selected_rows.at[][word_count]
        if row[2] < 8:
            pass
        elif w_c + row[2] < 100:
            w_c += row[2]
            single_string = " ".join(row[1])
            my_tuple[1].append(single_string)
    sum.append(my_tuple)

for i in sum:
    print (i)       
end_time = time.time()
cpu_time = end_time - start_time
print("CPU time in sec:", cpu_time)
