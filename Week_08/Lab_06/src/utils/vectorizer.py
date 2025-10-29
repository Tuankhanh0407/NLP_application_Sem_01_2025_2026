""" 
(i) Problem description:
This module implements TF-IDF vectorization for converting text into numerical features.

Input:
- List of tokenized texts.

Output:
- TF-IDF feature matrix.

(ii) Approach:
- Compute term frequency (TF) for each document.
- Compute inverse document frequency (IDF) across corpus.
- Calculate TF-IDF as TF * IDF.
"""

# Import necessary libraries 
from typing import List, Dict 
import numpy as np
from src.utils.tokenizer import RegexTokenizer

class TfidfVectorizer:
    """ 
    Convert text documents to TF-IDF feature vectors.

    Attributes:
    - vocab (dict): Mapping from terms to feature indices.
    - idf (np.array): Inverse document frequency vector.
    - tokenizer (RegexTokenizer): Text tokenizer.
    """

    def __init__(self, tokenizer: RegexTokenizer = None):
        """ 
        Initialize TF-IDF vectorizer.
        @param tokenizer (RegexTokenizer): Tokenizer instance. 
        """
        self.vocab = None 
        self.idf = None 
        self.tokenizer = tokenizer if tokenizer else RegexTokenizer()

    def build_vocabulary(self, texts: List[str]) -> None:
        """ 
        Build vocabulary from training texts.
        @param texts (List[str]): List of training documents. 
        """
        all_tokens = set()
        tokenized_texts = self.tokenizer.tokenize_batch(texts)

        for tokens in tokenized_texts:
            all_tokens.update(tokens)

        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """ 
        Learn vocabulary and transform texts to TF-IDF matrix.
        @param texts (List[str]): List of text documents.
        @return (np.array): TF-IDF feature matrix. 
        """
        self.build_vocabulary(texts)
        return self.compute_tfidf(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """ 
        Transform new texts to TF-IDF matrix using learned vocabulary.
        @param texts (List[str]): List of text documents.
        @return (np.array): TF-IDF feature matrix. 
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call fit_transform first.")
        return self.compute_tfidf(texts)
    
    def compute_tfidf(self, texts: List[str]) -> np.ndarray:
        """ 
        Compute TF-IDF matrix for input texts.
        @param texts (List[str]): List of text documents.
        @return (np.array): TF-IDF feature matrix. 
        """
        tokenized_texts = self.tokenizer.tokenize_batch(texts)
        n_docs = len(texts)
        n_features = len(self.vocab)

        # Compute term frequency (TF)
        tf_matrix = np.zeros((n_docs, n_features))
        for i, tokens in enumerate(tokenized_texts):
            total_terms = len(tokens)
            for token in tokens:
                if token in self.vocab:
                    tf_matrix[i, self.vocab[token]] += 1
            if total_terms > 0:
                tf_matrix[i] /= total_terms

        # Compute inverse document frequency (IDF)
        doc_freq = np.zeros(n_features)
        for i in range(n_features):
            doc_freq[i] = np.sum(tf_matrix[:, i] > 0)

        self.idf = np.log(n_docs / (doc_freq + 1))

        # Compute TF-IDF 
        tfidf_matrix = tf_matrix * self.idf 
        return tfidf_matrix 