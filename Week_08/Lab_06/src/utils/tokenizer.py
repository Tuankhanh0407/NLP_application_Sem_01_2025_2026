""" 
(i) Problem description:
This module implements text tokenization using regular expressions for splitting text into tokens.

Input:
- Raw text string.

Output:
- List of tokens (words).

(ii) Approach:
- Use regular expressions to split text into words.
- Convert to lowercase for normalization.
- Remove empty tokens.
"""

# Import necessary libraries
import re 
from typing import List

class RegexTokenizer:
    """ 
    Tokenize text using regular expressions.

    Attributes:
    - pattern (str): Regular expression pattern for tokenization. 
    """

    def __init__(self, pattern: str = r'\b\w+\b'):
        """ 
        Initialize tokenizer with specified pattern.
        @param pattern (str): Regular expression pattern for token matching.
        """
        self.pattern = pattern 

    def tokenize(self, text: str) -> List[str]:
        """ 
        Split text into tokens using regular expression.
        @param text (str): Input text to tokenize.
        @return (List[str]): List of extracted tokens. 
        """
        tokens = re.findall(self.pattern, text.lower())
        return [token for token in tokens if token.strip()] 
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """ 
        Tokenize multiple texts.
        @param texts (List[str]): List of input texts.
        @return (List[List[str]]): List of token lists for each text. 
        """
        return [self.tokenize(text) for text in texts]