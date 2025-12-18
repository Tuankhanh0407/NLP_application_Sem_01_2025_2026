# Import necessary libraries.
import spacy
from typing import List, Tuple

def extract_triplets(nlp: spacy.language.Language, text: str) -> List[Tuple[str, str, str]]:
    """ 
    Extract (subject, verb, object) triplets from a sentence.
    @param nlp (spacy.language.Language): The loaded spaCy pipeline.
    @param text (str): The sentence to analyze.
    @return (List[Tuple[str, str, str]]): List of found triplets as tuples (subject, verb, object). 
    """
    doc = nlp(text)
    triplets = []
    for token in doc:
        if token.pos_ == "VERB":
            verb = token.text
            subject = ""
            obj = ""
            for child in token.children:
                if child.dep_ == "nsubj":
                    subject = child.text
                if child.dep_ == "dobj":
                    obj = child.text
            if subject and obj:
                triplets.append((subject, verb, obj))
                print(f"Found triplet: ({subject}, {verb}, {obj})")
    if not triplets:
        print("No complete (subject, verb, object) triplets found.")
    return triplets

def extract_adjectives(nlp: spacy.language.Language, text: str) -> List[Tuple[str, List[str]]]:
    """ 
    Extract adjectives (amod) that modify nouns in a sentence.
    @param nlp (spacy.language.Language): The loaded spaCy pipeline.
    @param text (str): The sentence to analyze.
    @return (List[Tuple[str, List[str]]]): List of tuples (noun_text, [adjectives]). 
    """
    doc = nlp(text)
    results = []
    for token in doc:
        if token.pos_ == "NOUN":
            adjectives = []
            for child in token.children:
                if child.dep_ == "amod":
                    adjectives.append(child.text)
            if adjectives:
                results.append((token.text, adjectives))
                print(f"Noun '{token.text}' is modified by adjectives: {adjectives}")
    if not results:
        print("No noun-adjective relations (amod) found.")
    return results