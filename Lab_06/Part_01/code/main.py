""" 
(i) Problem description:
Implement three NLP tasks using transformer models:
1) Masked language modeling: Predict masked words in sentences.
2) Next token prediction: Generate text continuation.
3) Sentence representation: Convert sentences to vector embeddings.

Input:
- Various text inputs for different tasks.

Output:
- Predictions, generated text, and sentence embeddings.

(ii) Approach:
- Use Hugging Face transformers library for pre-trained models.
- Implement each task as seperate modules.
- Use appropriate models for each task (BERT for masked language modeling, GPT for text generation).
"""

# Import necessary libraries
import os
from src.task_01_masked_language_modeling import run_masked_language_modeling
from src.task_02_next_token_prediction import run_next_token_prediction
from src.task_03_sentence_representation import run_sentence_representation
import warnings

# Suppress unnecessary warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')

def main():
    """ 
    Main method to execute all NLP tasks. 
    """
    print("=== Lab 06 - Part 01 - Introduction to transformers ===")
    # Task 01: Masked language modeling
    print("Task 01: Masked language modeling")
    run_masked_language_modeling()
    print("\n" + "=" * 50 + "\n")
    # Task 02: Next token prediction
    print("Task 02: Next token prediction")
    run_next_token_prediction()
    print("\n" + "=" * 50 + "\n")
    # Task 03: Sentence representation
    print("Task 03: Sentence representation")
    run_sentence_representation()

if __name__ == "__main__":
    main()