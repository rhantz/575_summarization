from os import listdir, makedirs
from os.path import isfile, isdir, join, exists
from find_people import get_nlp, find_people
import sys


def convert_dict(ppl_dict):
    '''
    given Rachel's dict: dict[1st key: 'people'/'modified ppl']=[list of names/modified names]
    return a dict: new_dict[1st key: 'people'][2nd key: name] = [list of modified names in the original data]
    '''
    new_dict = {}
    new_dict['people'] = {}
    for i in range(len(ppl_dict['people'])):
        name = ppl_dict['people'][i]
        modified_name = ppl_dict["people_with_modifiers"][i].replace('\n', '')
        if name not in new_dict['people']:
            new_dict['people'][name] = [modified_name]
        else:
            new_dict['people'][name].append(modified_name)
    return new_dict


def track_topic(source_dir, prefix):
    '''
    given source directory and prefix D1001
    return path of topics eg: .../devtest/D1001A-A
    '''
    for file in listdir(source_dir):
        if isdir(join(source_dir, file)) and file.startswith(prefix):
            topic_path = join(source_dir, file)
    return topic_path


def find_longer_name(person, topic_dir):
    '''
    given a person's name in a summary
    find the longest name (with modifiers) from all articles under the topic_id where the summary is from
    '''
    name_string = ''  # initiate an empty string, concatenate any sent containing the name
    for article in listdir(topic_dir):
        with open(join(topic_dir, article), 'r') as f2:
            for l in f2:
                if person in l:
                    name_string += l
    ppl_dict_0 = convert_dict(find_people(name_string, nlp))
    # print('Names for *{}* under original Topic'.format(person), ppl_dict_0)
    modified = ppl_dict_0['people'][person]  # list of the names in the original article, with modifiers
    new_name = max(modified, key=len)
    return new_name


def generate_output(file_path, topic_dir):
    '''
    given a file path
    return the output to be written for D5
    '''
    with open(file_path, 'r') as f:
        pos_CL = ''  # initiate an empty output string
        names = set()
        for sent in f:
            ppl_dict = find_people(sent, nlp)

            # if there is no name, simply concatenate
            if ppl_dict['people'] == []:
                pos_CL += sent

            # if there are names, do the following
            else:
                # print('Rachel: ', sum, ppl_dict)
                for i in range(len(ppl_dict['people'])):
                    count_i = 0
                    person = ppl_dict['people'][i]
                    modified_person = ppl_dict['people_with_modifiers'][i]
                    new_person = find_longer_name(person, topic_dir).replace('\n', '')  # some names contain a \n
                    # print('Name change: ' + person + ' & ' + modified_person + ' --> ' + new_person + '\n')

                    # if a person's name appear for the first time and new name is longer
                    # replace it with the longer name
                    if person not in names:
                        names.add(person)
                        names.add(modified_person)
                        if len(new_person) >= len(modified_person):
                            new_sent = sent.replace(modified_person, new_person)
                        else:
                            new_sent = sent
                    else:
                        new_sent = sent.replace(modified_person, person)
                pos_CL += new_sent

    return pos_CL


if __name__ == "__main__":
    test_set = sys.argv[1]
    nlp = get_nlp()
    if test_set == 'dev':
        preCR_dir = f"../outputs/D5/before_CR/devtest/"
    else:
        preCR_dir = f"../outputs/D5/before_CR/evaltest/"

    summaries = sorted([f for f in listdir(preCR_dir) if isfile(join(preCR_dir, f))])
    for sum in summaries:
        if not sum.startswith('.'):  #ignore hidden '.DS_Store' file in MacOS
            file_path = join(preCR_dir, sum)
            topic_id = sum.split('-')[0]  # 'D1001' referring original data
            if test_set == 'dev':
                source_dir = f"../outputs/devtest"
            else:
                source_dir = f"../outputs/evaltest"
            topic_dir = track_topic(source_dir, topic_id)
            # print('\n\n' + topic_dir)
            pos_CL = generate_output(file_path, topic_dir)

            # create D5 dir and output files
            if test_set == 'dev':
                cr_dir = f"../outputs/D5_devtest"
            else:
                cr_dir = f"../outputs/D5_evaltest"
            if not exists(cr_dir):
                makedirs(cr_dir)
            D5_output = join(cr_dir, sum)
            with open(D5_output, 'w') as f2:
                f2.write(pos_CL)











