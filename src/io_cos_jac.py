import pdb
import sys
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
import numpy as np
from numpy.linalg import norm
import itertools
nltk.download('stopwords')
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
import export_summary

def read_filename(dir):
    '''
    Read file names from output/D3 directory
    '''
    fileNames = os.listdir(dir) #Get all files from outputs/D3 directory 
    return fileNames

def read_summary(dir, fileName):
    '''
    Read summaires 
    '''
    read_file = open(dir+'/'+fileName, "r") 
    data = read_file.read() 
    sentences = data.split('\n') 
    #Get rid of stopwords first.
    stop_word = set(stopwords.words('english'))
    word = data.lower().split(' ')
    s = []
    for i in range(len(word)):
        if (word[i] not in stop_word):
            s.append(word[i])

    s_w_s = " ".join(s)
    sentences_without_stopword  = s_w_s.split('\n')
    assert len(sentences_without_stopword) == len(sentences)
    return sentences, sentences_without_stopword

def cosine_sim(A,B):
    '''
    Compute cosine similarity between two word embeddings.
    '''
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine

def reorder(sentences, sentences_without_stopword):
    '''
    Reorder the summary based on cosine similarity. 
    Method c: Maximize Cosine Similarity
    Method j: Maximize Jaccard Similarity
    '''
    method = sys.argv[1]
    sentence_embeddings = sbert_model.encode(sentences_without_stopword)
    index = list(range(len(sentences)))
    #Find all combinations of sentence ordering. n! combinations in total.
    pers = list(itertools.permutations(index, len(index))) 
    if (method == 'c'):
        cos_dict = {}
        #Compute cosine similarities between two sentences.
        for s1 in range(len(sentences) - 1):
            for s2 in range(s1+1, len(sentences)):
                cos_dict[str(s1)+str(s2)] = cosine_sim(sentence_embeddings[s1], sentence_embeddings[s2])
        cos_count = {}
        for p in pers:
            total_cos = 0
            for s1 in range(len(p)-1):
                mylist = [p[s1], p[s1+1]]
                mylist.sort()
                total_cos += cos_dict[str(mylist[0])+str(mylist[1])]
            cos_count[str(p)] = total_cos
        best_per = max(cos_count, key=cos_count.get)
    elif (method == 'j'):
        #Compute Jaccard similarities between two sentences.
        jac_dict = {}
        for s1 in range(len(sentences) - 1):
            for s2 in range(s1+1, len(sentences)):
                intersection = set(sentences_without_stopword[s1].split(' ')) & set(sentences_without_stopword[s2].split(' '))
                union = set(sentences_without_stopword[s1].split(' ')).union(set(sentences_without_stopword[s2].split(' ')))
                jac_dict[str(s1)+str(s2)] = len(intersection) / len(union)
        jac_count = {}
        for p in pers:
            total_jac = 0
            for s1 in range(len(p)-1):
                mylist = [p[s1], p[s1+1]]
                mylist.sort()
                total_jac += jac_dict[str(mylist[0])+str(mylist[1])]
            jac_count[str(p)] = total_jac
        best_per = max(jac_count, key=jac_count.get)

    else:
        print("Incorrect method name.")
        pdb.set_trace()


    return best_per

def writeSummary(newOrder, dir, file):
    sentences, sentences_without_stopword = read_summary(dir, file)
    o = newOrder.strip('(').strip(')').split(',')
    assert len(sentences) == len(o)
    sent = []
    for i in range(len(o)):
        sent.append(sentences[int(o[i])])
    #topic_id D1001A
    topic_id = file[0:5]+file[-3]
    num = file[-1]
    export_dir = sys.argv[3]
    export_summary.export_summary(sent, topic_id, num+str(sys.argv[1]) , export_dir) #"../outputs/D4/A"
    

def main():
    dir = sys.argv[2] #'../outputs/D3'
    fileNames = read_filename(dir)
    for file in fileNames:
        sentences, sentences_without_stopword = read_summary(dir, file)
        if (len(sentences) > 1):
            newOrder = reorder(sentences, sentences_without_stopword)
            writeSummary(newOrder, dir, file)
        else:
            newOrder = '(0)'
        writeSummary(newOrder, dir, file)


if __name__ == "__main__":
   main()
