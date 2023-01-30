from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import pandas as pd
import export_summary
import sys
import string
import nltk
nltk.download('stopwords')


# Function to read in an article and return it as a single string
def read_article(file_path):

    with open(file_path, 'r') as file:
        return file.read()

def sent_score(doc_num, sent, tfidf_dic):
#     Purpose: To calc the tfidf score for each sentence 
#     Input: [doc_num]: index of the doc_set that the sentence is in
#            [sent]: a word_tokenized 
#            [tfidf_dic]: a dictionary with each word and their coresponding tfidf score
#     Output: The tfidf score of the sentence (taking average)
    score = 0
    word_num = 0
    for word in sent:
        if word.lower() in tfidf_dic:
            word_num += 1
            score += tfidf_dic[word.lower()][doc_num]
    if word_num > 0:
        return (score/word_num) 
    else:
        return 0

if __name__ == '__main__':

    directory = "../outputs/devtest"
    topic_ids = sorted([d for d in listdir(directory) if not isfile(join(directory, d))])
    total_top = len(topic_ids)
    all_set = []
    for topic_id in sorted(topic_ids):
        one_set = []
        topic_directory = f"../outputs/devtest/{topic_id}"
        articles = sorted([f for f in listdir(topic_directory) if isfile(join(topic_directory, f))])
        for name in articles:
            article_string = read_article(f"../outputs/devtest/{topic_id}/{name}")
            one_set.append(article_string)
            one_set_string = " ".join(one_set)
        all_set.append(one_set_string)


    stop_word = stopwords.words('english')
    # Create the TfidfVectorizer object
    # TfidfVectorizer only matches sequences of non-whitespace characters that are bounded by word boundaries 
    vectorizer = TfidfVectorizer(stop_words = stop_word, token_pattern=r'\b[^\s]+\b') #, ngram_range=(2, 2))

    # Fit and transform the documents to a matrix of TF-IDF features
    tfidf_matrix = vectorizer.fit_transform(all_set)

    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Make a dictionary with each word and their coresponding tfidf score
    tfidf_dict1 = dict(zip(feature_names, tfidf_matrix.toarray().T))

    sents = []
    word_count = []
    ave_score = []
    article_num = []
    word_count_no_stop = []

    # store informations in lists
    for iindex, doc in enumerate(all_set):
        sent_text = sent_tokenize(doc)
        for sentence in sent_text:
            para = sentence.split("\n\n")

            # filter out lines start with headline and date-time
            filtered_lines = [line for line in para if not line.startswith("headline:") and not line.startswith("date-time:")]
            filtered_text = " ".join(filtered_lines)
            tokens = word_tokenize(filtered_text)

            # filter out punctuation so when we count word, punctuation is excluded
            # filtered_token = [word for word in filtered_text if word not in string.punctuation]
            # filtered_token = word_tokenize(''.join(filtered_token))

            # filter out stop word so we can count word number exclude stop-word
            filtered_words = [word for word in tokens if word.lower() not in stop_word]

            # calculate sentence score 
            score = sent_score(iindex, tokens, tfidf_dict1)

            # creat list for panda df
            sents.append(filtered_text)
            word_count.append(len(tokens))
            word_count_no_stop.append(len(filtered_words))
            ave_score.append(score)
            article_num.append(iindex)

    # creates a DataFrame with four columns and populates each column with data from the variables 'article_num', 'sents', 'word_count', 'ave_score' respectively
    df = pd.DataFrame({'Doc_set':article_num, 'sentence': sents, 'word_count_no_stop': word_count_no_stop, 'word_count': word_count, 'tfidf_scores': ave_score})
    df = df.sort_values(by = ['Doc_set', 'tfidf_scores'], ascending = [True, False])
    df = df.reset_index()
    df.drop(df.columns[0], axis=1, inplace=True)

    # print(df.to_string())    

    cosine_sim = float(sys.argv[1])
    word_co = int(sys.argv[2])
    total_set = iindex + 1
    summary = []
    # select sentences with high tfidf scores 
    for top_num in range(0, total_set):
        my_tuple = (topic_ids[top_num], [])
        w_c = 0
        
        # iterate through dataframe where Doc_set = top_num, so that we are only selecting sentences in the same doc_set
        for index, row in df.loc[df['Doc_set'] == top_num].iterrows():

            # filter out sentence with less than 8 words 
            if row[2] < word_co:
                pass

            else:
                if len(my_tuple[1]) == 0:
                    first_string = row[1].strip('\n')
                    my_tuple[1].append(first_string)
                    w_c += row[3]
                    continue

                v_curr = vectorizer.transform([row[1]])
                total_cos_simi = 0

                # check similarity bwtween sentences using cosine_similarity, filter out sentences with similar meaning
                for sent in my_tuple[1]:
                    v_selected = vectorizer.transform([sent])
                    total_cos_simi += cosine_similarity(v_curr, v_selected)
                avg_cos_simi = total_cos_simi / len(my_tuple[1])
                if  avg_cos_simi > cosine_sim:
                    continue
                # else:
                    # make sure summary does not exceed 100 words limit
                if w_c + row[2] < 100:
                    single_string = row[1].strip('\n')
                    my_tuple[1].append(single_string)
                    w_c += row[3]
                else:
                    break
        summary.append(my_tuple)

    # export the result
    for top_id, sent in summary:
        export_summary.export_summary(sent, top_id[:6], "1", "../outputs/D3")