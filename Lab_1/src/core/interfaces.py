# Import necessary libraries.
from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    """ 
    Abstract base class for tokenizers. 
    """
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """ 
        Tokenize input text into a list of tokens.
        @param text (str): Input text to tokenize.
        @return (List[str]): List of tokens extracted from text. 
        """
        pass

class Vectorizer(ABC):
    """ 
    Abstract base class for vectorizers. 
    """
    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """ 
        Learn vocabulary from a corpus.
        @param corpus (List[str]): List of documents to learn vocabulary from. 
        """
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """ 
        Transform documents into vectors based on learned vocabulary.
        @param documents (List[str]): Documents to transform.
        @return (List[List[int]]): Document-term count vectors. 
        """
        pass

    @abstractmethod
    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """ 
        Convenience method to fit and transform the same corpus.
        @param corpus (List[str]): Corpus to fit and transform.
        @return (List[List[int]]): Document-term count vectors. 
        """
        pass