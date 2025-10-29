""" 
(i) Problem description:
This module implements a text classifier using logistic regression for binary classification.

Input:
- Text documents and labels.

Output:
- Trained classification model and predictions.

(ii) Approach:
- Use TF-IDF for text vectorization.
- Apply logistic regression for classification.
- Evaluate using standard metrics.
"""

# Import necessary libraries
from typing import List, Dict 
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from src.utils.vectorizer import TfidfVectorizer

class TextClassifier:
    """ 
    Text classification model using logistic regression.

    Attributes:
    - vectorizer (TfidfVectorizer): Text vectorization component.
    - model (LogisticRegression): Trained classification model. 
    """

    def __init__(self, vectorizer: TfidfVectorizer):
        """ 
        Initialize text classifier with vectorizer.
        @param vectorizer (TfidfVectorizer): Vectorizer for text processing.  
        """
        self.vectorizer = vectorizer 
        self.model = None 

    def fit(self, texts: List[str], labels: List[int]) -> None:
        """ 
        Train the classifier on text data.
        @param texts (List[str]): Training text documents.
        @param labels (List[int]): Training labels (0 or 1). 
        """
        # Transform texts to features
        X = self.vectorizer.fit_transform(texts)

        # Initialize and train logistic regression 
        self.model = LogisticRegression(solver = 'liblinear', random_state = 42)
        self.model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """ 
        Predict labels for new text documents.
        @param texts (List[str]): Text documents to classify.
        @return (List[int]): Predicted labels (0 or 1). 
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() method first.")
        
        X = self.vectorizer.transform(texts)
        return self.model.predict(X).tolist()
    
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """ 
        Calculate evaluation metrics for predictions.
        @param y_true (List[int]): True labels.
        @param y_pred (List[int]): Predicted labels.
        @return (Dict[str, float]): Dictionary of evaluation metrics.
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division = 0),
            'recall': recall_score(y_true, y_pred, zero_division = 0),
            'f1_score': f1_score(y_true, y_pred, zero_division = 0)
        }