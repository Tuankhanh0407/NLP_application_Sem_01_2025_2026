""" 
(i) Problem description:
Test script for improved text classification using Naive Bayes and enhanced preprocessing.

Input:
- Sample movie review texts and labels.

Output:
- Improved model performance metrics.

(ii) Approach:
- Implement Naive Bayes classifier.
- Add advanced preprocessing (stop words removal, punctuation handling).
- Compare with baseline performance.
"""

# Import necessary libraries
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import re 
from typing import List 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from src.utils.tokenizer import RegexTokenizer
from src.utils.vectorizer import TfidfVectorizer

class ImprovedTokenizer(RegexTokenizer):
    """ 
    Enhanced tokenizer with stop words removal and better text cleaning.

    Attributes:
    - stop_words (set): Set of common stop words to remove. 
    """

    def __init__(self):
        """ 
        Initialize improved tokenizer with stop words. 
        """
        super().__init__(r'\b[a-zA-Z]{2,}\b') # Only words with 2+ letters
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that',
            'it', 'its', 'it\'s', 'i', 'you', 'he', 'she', 'we', 'they', 'my'
        }

    def tokenize(self, text: str) -> List[str]:
        """ 
        Tokenize text with enhanced preprocessing.
        @param text (str): Input text. 
        @return (List[str]): Cleaned tokens. 
        """
        # Remove punctuation and convert to lowercase 
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = super().tokenize(text)
        # Remove stop words and short tokens
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]
    
class ImprovedTextClassifier:
    """ 
    Improved text classifier using Naive Bayes.

    Attributes:
    - vectorizer (TfidfVectorizer): Enhanced vectorizer.
    - model (MultinomialNB): Naive Bayes model. 
    """

    def __init__(self, vectorizer: TfidfVectorizer):
        """ 
        Initialize improved classifier.
        @param vectorizer (TfidfVectorizer): Enhanced vectorizer.
        """
        self.vectorizer = vectorizer 
        self.model = None 

    def fit(self, texts: List[str], labels: List[int]) -> None:
        """ 
        Train the improved classifier.
        @param texts (List[str]): Training texts.
        @param labels (List[int]): Training labels. 
        """
        X = self.vectorizer.fit_transform(texts)
        self.model = MultinomialNB()
        self.model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """ 
        Predict labels with improved model.
        @param texts (List[str]): Texts to classify.
        @return (List[int]): Predicted labels. 
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() method first.")
        
        X = self.vectorizer.transform(texts)
        return self.model.predict(X).tolist()
    
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> dict:
        """ 
        Evaluate model performance.
        @param y_true (List[int]): True labels.
        @param y_pred (List[int]): Predicted labels.
        @return (dict): Evaluation metrics. 
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division = 0),
            'recall': recall_score(y_true, y_pred, zero_division = 0),
            'f1_score': f1_score(y_true, y_pred, zero_division = 0)
        }
    
def main():
    """ 
    Main function to run improved text classification test. 
    """
    print("=== Improved text classification test ===")

    # Extended dataset for better evaluation
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommnend this, a masterpiece.",
        "Could not finish watching, so bad.",
        "Excellent cinematography and compelling story.",
        "Poor script and weak character development.",
        "One of the best movies I've ever seen!",
        "Disappointing and poorly executed.",
        "Brilliant performance by all actors involved.",
        "The plot was predictable and uninspired.",
        "A must-see for all cinema lovers!",
        "I regret spending money on this movie."
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    print(f"Dataset: {len(texts)} texts")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size = 0.3, random_state = 42, stratify = labels 
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Initialize improved components 
    print("\nInitializing improved tokenizer and vectorizer...")
    tokenizer = ImprovedTokenizer()
    vectorizer = TfidfVectorizer(tokenizer)
    classifier = ImprovedTextClassifier(vectorizer)

    # Train improved classifier
    print("Training improved classifier (Naive Bayes)...")
    classifier.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_pred = classifier.predict(X_test)

    # Evaluate improved model 
    metrics = classifier.evaluate(y_test, y_pred)

    print("\n=== Improved model results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")

    # Show tokenization improvement example 
    print("\n=== Preprocessing improvement example ===")
    sample_text = "This movie is really great and I love it, but it's too long!"
    print(f"Original text: '{sample_text}'")
    print(f"Improved tokens: {tokenizer.tokenize(sample_text)}")

if __name__ == "__main__":
    main()