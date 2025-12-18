# Import necessary libraries.
import spacy
from typing import List

def format_children(token) -> List[str]:
    """ 
    Return a list of child token texts for a given value.
    @param token (spacy.tokens.Token): Token whose children are to be listed.
    @return (List[str]): List of child token texts. 
    """
    return [child.text for child in token.children]

def print_token_info(nlp: spacy.language.Language, text: str) -> None:
    """ 
    Parse a sentence and print token attributes in a formatted table.
    @param nlp (spacy.language.Language): The loaded spaCy pipeline.
    @param text (str): The sentence to parse. 
    """
    doc = nlp(text)
    header = f"{'TEXT':<12} | {'DEP':<10} | {'HEAD TEXT':<12} | {'HEAD POS':<8} | CHILDREN"
    print("\n" + header)
    print("-" * len(header))
    for token in doc:
        children = format_children(token)
        print(f"{token.text:<12} | {token.dep_:<10} | {token.head.text:<12} | {token.head.pos_:<8} | {children}")