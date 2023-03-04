from os import listdir, makedirs
from os.path import isfile, isdir, join, exists
from find_people import get_nlp, find_people


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
    given source directory devtest and prefix D1001
    return path of topics .../devtest/D1001A-A
    '''
    for file in listdir(source_dir):
        if isdir(join(source_dir, file)) and file.startswith(prefix):
            topic_path = join(source_dir, file)
    return topic_path


def find_longer_name(person):
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


def generate_output(D4_filename):
    '''
    given a file name in D4
    return the output to be written for D5
    '''
    with open(f"../outputs/D4/{D4_filename}", 'r') as f:
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
                    new_person = find_longer_name(person).replace('\n', '')  # some names contain a \n
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
    nlp = get_nlp()
    outputs_dir = f"../outputs/D4"
    summaries = sorted([f for f in listdir(outputs_dir) if isfile(join(outputs_dir, f))])
    for sum in summaries:
        if not sum.startswith('.'):  #ignore hidden '.DS_Store' file in MacOS
            topic_id = sum.split('-')[0]  # 'D1001' referring original data
            source_dir = f"../outputs/devtest"
            topic_dir = track_topic(source_dir, topic_id)
            # print('\n\n' + topic_dir)
            pos_CL = generate_output(sum)

            # create D5 dir and output files
            D5 = f"../outputs/D5"
            if not exists(D5):
                makedirs(D5)
            D5_output = join(D5, sum)
            with open(D5_output, 'w') as f2:
                f2.write(pos_CL)











