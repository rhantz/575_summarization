import spacy
from spacy import Language


def get_nlp() -> Language:
    """
    Loads nlp model for finding people and their modifiers.

    e.g. Run:
    nlp = get_nlp()

    Returns:
        nlp model

    """
    return spacy.load("en_core_web_sm", disable=["lemmatizer"])


def find_people(sent, model) -> dict:
    """
    Given a sentence, performs Named Entity Recognition and Dependency Parsing to retrieve:
        list of people in sentence
        list of people (modifiers included) in sentence

    Args:
        sent: string sentence
        model: nlp model (retrieved using get_nlp())

    Returns:
        people_dict:
            {
                "people": list of people in sentence
                "people_with_modifiers": list of people (modifiers included) in sentence
            }

    """

    doc = model(sent)

    people = [entity.text for entity in doc.ents if entity.label_ == 'PERSON']

    people_tokens = " ".join(people).split()  # all tokens for people

    only_people_with_modifiers = []  # list for only people with modifiers

    for noun_phrase in doc.noun_chunks:
        noun_head = noun_phrase.root

        # checks that the head of this noun phrase was found in people tokens
        if noun_head.text not in set(people_tokens):
            continue
        # only stores noun phrases that are people with modifiers
        if noun_phrase.text not in people:  # Since, people list does not include modifiers
            only_people_with_modifiers.append(noun_phrase.text)

    people_with_modifiers = []  # list for all people, including those with modifiers

    for person in people:

        # selects all modified versions of that person
        modified_person = [
            modified_person for modified_person in only_people_with_modifiers if person in modified_person
        ]

        if modified_person:
            # appends first modified person
            people_with_modifiers.append(modified_person[0])
            # removes that modified person so that it is not appended again
            only_people_with_modifiers.remove(modified_person[0])
        else:
            # if there was never a modifier for that person, just appends un-modified person
            people_with_modifiers.append(person)

    return {
        "people": people,
        "people_with_modifiers": people_with_modifiers
    }


if __name__ == "__main__":

    # Get nlp model (you should only run this line once)
    nlp = get_nlp()

    sentence = "President Barack Obama has two daughters, Malia and Sasha Obama, and his wife is First Lady Michelle Obama."

    # for each sentence, run "find_people" to get a people_dict (Schema shown below in example output)
    people_dict = find_people(sentence, nlp)

    """
    e.g. Output of people_dict
    Note. "PERSON" entities are retrieved via SpaCy's named entity recognizer model, so some errors in prediction
    can be expected ("Malia" is a False Negative)
    
    {
        # 'people' is a list of names, no modifiers, in the order they appear in the sentence
        'people': ['Barack Obama', 'Sasha Obama', 'Michelle Obama'],
        
        # 'people_with_modifiers' is a list of names, with any modifiers, in the order they appear in the sentence 
        'people_with_modifiers': ['President Barack Obama', 'Sasha Obama', 'First Lady Michelle Obama']
    }
    """
