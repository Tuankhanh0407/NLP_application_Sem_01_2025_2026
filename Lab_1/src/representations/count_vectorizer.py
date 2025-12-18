# Import necessary libraries.
import numpy
from src.core.interfaces import Vectorizer, Tokenizer
from typing import Dict, List

class CountVectorizer(Vectorizer):
    """ 
    Simple bag-of-words count vectorizer.

    Attributes:
    - tokenizer (Tokenizer): Tokenizer instance used to split documents.
    - vocab (Dict[str, int]): Mapping from token to index. 
    """
    def __init__(self, tokenizer: Tokenizer):
        """ 
        Initialize the CountVectorizer class. 
        """
        self.tokenizer = tokenizer
        self.vocab: Dict[str, int] = {}

    def fit(self, corpus: List[str]) -> None:
        """ 
        Learn vocabulary from the corpus using the tokenizer.
        @param corpus (List[str]): List of documents to learn from. 
        """
        unique_tokens = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)
        # Create deterministic ordering by sorting tokens.
        sorted_tokens = sorted(unique_tokens)
        self.vocab = {token: idx for idx, token in enumerate(sorted_tokens)}

    def transform(self, documents: List[str]) -> List[List[int]]:
        """ 
        Transform documents into count vectors based on learned vocabulary.
        @param documents (List[str]): Documents to transform.
        @return (List[List[int]]): List of integer count vectors. 
        """
        if not self.vocab:
            raise ValueError("Vocabulary is empty. Call fit() before transform().")
        vocab_size = len(self.vocab)
        vectors = []
        for doc in documents:
            vec = [0] * vocab_size
            tokens = self.tokenizer.tokenize(doc)
            for t in tokens:
                if t in self.vocab:
                    vec[self.vocab[t]] += 1
            vectors.append(vec)
        return vectors
    
    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """ 
        Convenience method to fit and transform the same corpus.
        @param corpus (List[str]): Corpus to fit and transform.
        @return (List[List[int]]): Document-term count vectors. 
        """
        self.fit(corpus)
        return self.transform(corpus)