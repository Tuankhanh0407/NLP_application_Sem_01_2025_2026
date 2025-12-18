# Import necessary libraries
import re
from src.core.interfaces import Tokenizer
from typing import List

class SimpleTokenizer(Tokenizer):
    """ 
    A very simple tokenizer that lowercases text, seperates basic punctuation, and splits on whitespace.

    Attributes:
    - punctuation_chars (str): String of punctuation characters to seperate. 
    """
    def __init__(self):
        """ 
        Initialize the SimpleTokenizer class. 
        """
        self.punctuation_chars = r'\.\,\?\!\:\;\(\)\[\]\"\''
        # Precompile regular expression to add spaces around punctuation.
        self.punctuation_regex = re.compile(r'([.,?!:;()\[\]"\'])')

    def tokenize(self, text: str) -> List[str]:
        """ 
        Tokenize text by lowercasing, separating punctuation, and splitting on whitespace.
        @param text (str): Input text to tokenize.
        @return (List[str]): List of tokens. 
        """
        if text is None:
            return []
        # Lowercase.
        lowered = text.lower()
        # Put spaces around punctuation so they become seperate tokens.
        spaced = self.punctuation_regex.sub(r' \1 ', lowered)
        # Normalize whitespace and split.
        tokens = spaced.split()
        return tokens