""" 
(i) Problem description:
Test script for baseline text classification pipeline using logistic regression.

Input:
- Sample movie review texts and labels.

Output:
- Model performance metrics.

(ii) Approach:
- Split data into train/test sets.
- Train logistic regression classifier.
- Evaluate on test set.
- Print performance metrics. 
"""

# Import necessary libraries
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.model_selection import train_test_split 
from src.utils.tokenizer import RegexTokenizer
from src.utils.vectorizer import TfidfVectorizer
from src.models.text_classifier import TextClassifier

def main():
    """ 
    Main function to run baseline text classification test. 
    """
    print("=== Baseline text classification test ===")

    # Define dataset
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad.",
        "Excellent cinematography and compelling story.",
        "Poor script and weak character development."
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0] # 1 = positive, 0 = negative

    print(f"Dataset: {len(texts)} texts with {sum(labels)} positive and {len(labels) - sum(labels)} negative")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size = 0.25, random_state = 42, stratify = labels 
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Initialize components 
    tokenizer = RegexTokenizer()
    vectorizer = TfidfVectorizer(tokenizer)
    classifier = TextClassifier(vectorizer)

    # Train classifier
    print("\nTraining classifier...")
    classifier.fit(X_train, y_train)

    # Make predictions 
    print("Making predictions...")
    y_pred = classifier.predict(X_test)

    # Evaluate 
    metrics = classifier.evaluate(y_test, y_pred)

    print("\n=== Evaluation results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")

    # Show test examples 
    print("\n=== Test examples ===")
    for i, (text, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):
        sentiment_true = "positive" if true_label == 1 else "negative"
        sentiment_pred = "positive" if pred_label == 1 else "negative"
        status = "CORRECT" if true_label == pred_label else "WRONG"
        print(f"Example {i + 1}: True = {sentiment_true}, Predicted = {sentiment_pred}, Result = {status}")
        print(f"    Text: '{text}'")

if __name__ == "__main__":
    main()