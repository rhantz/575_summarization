from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import defaultdict
import pandas as pd
import export_summary
import sys
import argparse
import nltk
nltk.download('stopwords')

stop_word = stopwords.words('english')
def order_info(index,sents,do_order):
#     Purpose: To order sentences in the summary 
#     Input: [index]: index of the doc_set that the sentence is in
#            [sent]: a sentence list
#            [do_order]: optioin to perform information order or not
#     Output: topic_ID and the summary (list)

    pos = {}
    for i in sents:
        pos[int(i[1])/int(i[0])] = i[2]
    if do_order == "yes": 
        sorted_pos = dict(sorted(pos.items()))
        pos_list=list(sorted_pos.values())
    else:
        pos_list = list(pos.values())


    return(index,pos_list)


def is_valid_sentence(sentence):
    return sentence != '' and not sentence.startswith("headline: ") and not sentence.startswith("date-time: ")

# Function to read in an article and return it as a single string
def read_article(file_path):

    with open(file_path, 'r') as file:
        return file.read()

def sent_score(doc_num, sent, tfidf_dic, gram_type):
#     Purpose: To calc the tfidf score for each sentence 
#     Input: [doc_num]: index of the doc_set that the sentence is in
#            [sent]: a word_tokenized 
#            [tfidf_dic]: a dictionary with each word and their coresponding tfidf score
#     Output: The tfidf score of the sentence (taking average)
    score = 0
    word_num = 0
    if gram_type == "bigram":
        for word in sent:
            if word in tfidf_dic:
                word_num += 1
                score += tfidf_dic[word][doc_num]
        if word_num > 0:
            return (score/word_num) 
        else:
            return 0
            
    elif gram_type == "unigram":
        for word in sent:
            if word.lower() in tfidf_dic:
                # if word.lower() not in stop_word:
                word_num += 1
                score += tfidf_dic[word.lower()][doc_num]
        if word_num > 0:
            return (score/word_num) 
        else:
            return 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="input directory to preprocessed articles (training, devtest, evaltest)")
    parser.add_argument(
        "--cosine_sim", type=float, required=True, help="cosine similiar to select summary")
    parser.add_argument(
        "--word_co", type=int, required=True, help="word limit to exclude sentences under certain word boundry")
    parser.add_argument(
        "--gram_type", type=str, required=True, help="unigram/bigram")
    parser.add_argument(
        "--info_order", type=str, required=True, help="option to perform info order (yes/no)")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="output directory to print summary to")

    args = parser.parse_args(sys.argv[1:])
    directory = f"../outputs/{args.input_dir}/"

    topic_ids = sorted([d for d in listdir(directory) if not isfile(join(directory, d))])
    total_top = len(topic_ids)
    all_set = []
    all_sent = []
    for topic_id in sorted(topic_ids):
        one_set = []
        topic_directory = f"../outputs/{args.input_dir}/{topic_id}"
        articles = sorted([f for f in listdir(topic_directory) if isfile(join(topic_directory, f))])
        for name in articles:
            article_string = read_article(f"../outputs/{args.input_dir}/{topic_id}/{name}")
            sent_string = article_string.split('\n')
            one_set.append(article_string)
            one_set_string = " ".join(one_set)
            all_sent.append(sent_string)
        all_set.append(one_set_string)


    # Create the TfidfVectorizer object
    # TfidfVectorizer only matches sequences of non-whitespace characters that are bounded by word boundaries 

    # unigram
    if args.gram_type == "unigram":
        vectorizer = TfidfVectorizer(stop_words = stop_word, token_pattern=r'\b[^\s]+\b')

    # bigram
    if args.gram_type == "bigram":
        vectorizer = TfidfVectorizer(token_pattern=r'\b[^\s]+\b', ngram_range=(2, 2))

    # Fit and transform the documents to a matrix of TF-IDF features
    tfidf_matrix = vectorizer.fit_transform(all_set)

    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Make a dictionary with each word and their coresponding tfidf score
    tfidf_dict1 = dict(zip(feature_names, tfidf_matrix.toarray().T))
    tokenizer = nltk.RegexpTokenizer(r'\b[^\s]+\b')

    sent_data = defaultdict(list)
    for article_index, sentences in enumerate(all_sent):
        sentence_index = 0
        for sentence in sentences:
            if is_valid_sentence(sentence):
                tokens = word_tokenize(sentence)
                if args.gram_type == "bigram":
                    # tokenizer = nltk.RegexpTokenizer(r'\b[^\s]+\b')
                    tokens_for_bi = tokenizer.tokenize(sentence)
                    bigrams = list(nltk.bigrams(tokens_for_bi))
                    for i in range(len(bigrams)):
                        bigrams[i] = bigrams[i][0].lower() + ' ' + bigrams[i][1].lower()
                    score = sent_score(article_index // 10, bigrams, tfidf_dict1, args.gram_type)
                if args.gram_type == "unigram":
                    score = sent_score(article_index // 10, tokens, tfidf_dict1, args.gram_type)
                filtered_tokens = [word for word in tokens if word.lower() not in stop_word]
                sent_data["sentence"].append(sentence)
                sent_data["word_count"].append(len(tokens))
                sent_data["word_count_no_stop"].append(len(filtered_tokens))
                sent_data["ave_score"].append(score)
                sent_data["topic_id"].append(article_index // 10)
                sent_data["sentence_index"].append(sentence_index)
                sent_data["article_index"].append(article_index%10)
                sentence_index += 1
        sent_data["num_sent"].extend([sentence_index]*(sentence_index))


    # creates a DataFrame with four columns and populates each column with data from the variables 'article_num', 'sents', 'word_count', 'ave_score' respectively
    df = pd.DataFrame({'Topic_id':sent_data["topic_id"],'Article_id': sent_data["article_index"], 'sentence': sent_data["sentence"], 'word_count_no_stop': sent_data["word_count_no_stop"], 'word_count': sent_data["word_count"], 'tfidf_scores': sent_data["ave_score"], 'sent_index':sent_data["sentence_index"], 'num_sent':sent_data["num_sent"]})
    df = df.sort_values(by = ['Topic_id', 'tfidf_scores'], ascending = [True, False])
    df = df.reset_index()
    df.drop(df.columns[0], axis=1, inplace=True)
 
    # print(df.to_string()) 

    total_set = int((article_index + 1)/10)
    summary = []
    # select sentences with high tfidf scores 
    for top_num in range(0, total_set):
        curr_topID = topic_ids[top_num]
        info_order = []
        w_c = 0
        
        # iterate through dataframe where Doc_set = top_num, so that we are only selecting sentences in the same doc_set
        for index, row in df.loc[df['Topic_id'] == top_num].iterrows():
            # filter out sentence with less than 8 words 
            num_tokens = tokenizer.tokenize(row[2])
            num_punct = row[4] - len(num_tokens)
            # filter out sentence with less than 8 words 
            if (row[4]-num_punct) < args.word_co:
                pass

            else:
                if len(info_order) == 0:
                    first_string = row[2].strip('\n')
                    info_order.append([str(row[-1]),str(row[-2]),first_string])
                    w_c += row[4]
                    continue

                v_curr = vectorizer.transform([row[2]])
                total_cos_simi = 0

                # check similarity bwtween sentences using cosine_similarity, filter out sentences with similar meaning
                for sent in info_order:
                    v_selected = vectorizer.transform([sent[2]])
                    total_cos_simi += cosine_similarity(v_curr, v_selected)
                avg_cos_simi = total_cos_simi / len(info_order)
                if  avg_cos_simi > args.cosine_sim:
                    continue
                # else:
                    # make sure summary does not exceed 100 words limit
                if w_c + row[4] < 100:
                    single_string = row[2].strip('\n')
                    info_order.append([str(row[-1]),str(row[-2]),single_string])
                    w_c += row[4]
                else:
                    continue
        
        summary.append(order_info(curr_topID,info_order,args.info_order))

    # # # export the result
    for top_id, sent in summary:
        export_summary.export_summary(sent, top_id[:6], "1", args.output_dir)