""" 
(i) Problem description:
Predict masked words in sentences using masked language modeling.

Input:
- Sentence with [MASK] token.

Output:
- Top predictions for the masked word with confidence scores.

(ii) Approach:
- Use BERT-based model through Hugging Face pipeline.
- Leverage encoder-only architecture for bidirectional context understanding.
"""

# Import necessary libraries
from transformers import pipeline

class MaskedLanguageModeling:
    """ 
    Perform masked language modeling tasks.
    """

    def __init__(self):
        """ 
        Initialize the masked language modeling pipeline. 
        """
        self.mask_filler = pipeline("fill-mask", model = "distilbert-base-uncased")

    def predict_masked_tokens(self, sentence, top_k = 5):
        """ 
        Predict masked tokens in a sentence.
        @param sentence (str): Input sentence with [MASK] token.
        @param top_k (int): Number of top predictions to return.
        @return (list): List of predictions with tokens and scores.
        """
        predictions = self.mask_filler(sentence, top_k = top_k)
        return predictions
    
    def display_predictions(self, original_sentence, predictions):
        """ 
        Display predictions in a formatted way.
        @param original_sentence (str): Original input sentence.
        @param predictions (list): List of prediction dictionaries.
        """
        print(f"Original sentence: {original_sentence}")
        print("Top predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. Token: '{pred['token_str']}' | Confidence: {pred['score']:.4f}")
            print(f"    Complete sentence: {pred['sequence']}")

def run_masked_language_modeling():
    """ 
    Execute masked language modeling task. 
    """
    try:
        mlm = MaskedLanguageModeling()
        # Test sentence
        input_sentence = "Hanoi is the [MASK] of Vietnam."
        # Get predictions
        predictions = mlm.predict_masked_tokens(input_sentence, top_k = 5)
        # Display results
        mlm.display_predictions(input_sentence, predictions)
    except Exception as e:
        print(f"Error in masked language modeling: {e}")