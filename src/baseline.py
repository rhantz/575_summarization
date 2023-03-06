import pdb
from os import listdir
from os.path import isfile, join
import export_summary
import sys

# Function to read in an article and return it as a single string
def read_article(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def getSummary(article_string):
    para = article_string.split("\n\n")
    filtered_lines = [line for line in para if not line.startswith("headline:") and not line.startswith("date-time:")]
    word_count = 0  
    word_limit = 100
    result = []
    for i in filtered_lines:
        if (i != ''):
            if (len(i.strip(' ').split(' ')) + word_count <= word_limit):
                result.append(i.strip('\n'))
                word_count += len(i.split(' '))
            else:
                break

    return result

def main():
    directory = sys.argv[1] #"../outputs/devtest"
    topic_ids = sorted([d for d in listdir(directory) if not isfile(join(directory, d))])
    total_top = len(topic_ids)
    all_set = []
    export_dir = sys.argv[2]
    for topic_id in sorted(topic_ids):
        one_set = []
        topic_directory =  directory+'/'+str(topic_id) #f"../outputs/devtest/{topic_id}"
        articles = sorted([f for f in listdir(topic_directory) if isfile(join(topic_directory, f))])
        # article_string = read_article(f"../outputs/devtest/{topic_id}/"+articles[0])    
        article_string = read_article(topic_directory + "/" + articles[0])      
        summary = getSummary(article_string)
        export_summary.export_summary(summary, topic_id[:6], "3", export_dir) #"../outputs/D4"

if __name__ == '__main__':
    main()

