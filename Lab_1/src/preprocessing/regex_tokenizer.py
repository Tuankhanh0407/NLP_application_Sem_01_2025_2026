# Import necessary libraries.
import re
from src.core.interfaces import Tokenizer
from typing import List

class RegexTokenizer(Tokenizer):
    """ 
    Tokenizer that uses a single regular expression to extract tokens.

    Attributes:
    - token_pattern (re.Pattern): Compiled regular expression pattern used to extract tokens. 
    """
    def __init__(self):
        """ 
        Initialize the RegexTokenizer class. 
        """
        # Pattern involving words (including digits and underscores) OR any single non-whitespace non-word character (punctuation).
        self.token_pattern = re.compile(r'\w+|[^\w\s]', re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        """ 
        Tokenize text using a regular expression that extracts words and punctuation.
        @param text (str): Input text to tokenize.
        @return (List[str]): List of tokens (lowercased). 
        """
        if text is None:
            return []
        matches = self.token_pattern.findall(text)
        # Normalize to lowercase for words; punctuation remains as-is but lower() is safe.
        tokens = [m.lower() for m in matches]
        return tokens