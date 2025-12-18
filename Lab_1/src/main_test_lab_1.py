""" 
(i) Problem description:
Implement and demonstrate tokenizers (simple and regular expression) and a CountVectorizer.

Input:
- Predefined test sentences and a sample dataset (embedded or loaded).

Output:
- Printed tokenization results and document-term matrices.

(ii) Approach:
- Organize program into many small helper functions.
- Provide clear printed outputs for evaluation.
"""

# Import necessary libraries.
import os
import pprint
from src.core.dataset_loaders import load_raw_text_data
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List

def prepare_test_sentences() -> List[str]:
    """ 
    Prepare the list of sentences used for testing tokenizers.
    @return (List[str]): List of test sentences. 
    """
    return [
        "Hello, world! This is a test.",
        "NLP is fascinating ... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

def demo_tokenizers(simple_tokenizer: SimpleTokenizer, regex_tokenizer: RegexTokenizer, sentences: List[str]) -> None:
    """ 
    Demonstrate tokenization on a list of sentences and print results.
    @param simple_tokenizer (SimpleTokenizer): Instance of SimpleTokenizer.
    @param regex_tokenizer (RegexTokenizer): Instance of RegexTokenizer.
    @param sentences (List[str]): Sentences to tokenize. 
    """
    print("\n--- Tokenizer demonstration ---")
    for s in sentences:
        print(f"Original: {s}")
        stoks = simple_tokenizer.tokenize(s)
        rtoks = regex_tokenizer.tokenize(s)
        print(f"SimpleTokenizer: {stoks}")
        print(f"RegexTokenizer: {rtoks}")
        print()

def demo_count_vectorizer(tokenizer: RegexTokenizer, corpus: List[str]) -> None:
    """ 
    Demonstrate CountVectorizer using the provided tokenizer and corpus.
    @param tokenizer (RegexTokenizer): Tokenizer instance to use.
    @param corpus (List[str]): Corpus of documents. 
    """
    print("\n--- CountVectorizer demonstration ---")
    vectorizer = CountVectorizer(tokenizer)
    dt_matrix = vectorizer.fit_transform(corpus)
    print("Learned vocabulary (token -> index):")
    pprint.pprint(vectorizer.vocab)
    print("\nDocument-term matrix (rows = documents)")
    for i, vec in enumerate(dt_matrix):
        print(f"Doc {i}: {vec}")

def demo_dataset_tokenization(simple_tokenizer: SimpleTokenizer, regex_tokenizer: RegexTokenizer, dataset_path: str = None) -> None:
    """ 
    Load a sample of raw text data and tokenize it with both tokenizers.
    @param simple_tokenizer (SimpleTokenizer): Simple tokenizer instance.
    @param regex_tokenizer (RegexTokenizer): Regular expression tokenizer instance.
    @param dataset_path (str): Optional dataset path to load real data. 
    """
    raw_text = load_raw_text_data(dataset_path)
    sample_text = raw_text[:500] # First 500 characters for demonstration.
    print("\n--- Tokenizing sample text from dataset (first 100 chars shown) ---\n")
    print(f"Origina; sample (first 100 chars): {sample_text[:100]}...\n")
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer output (first 20 tokens): {simple_tokens[:20]}")
    print(f"RegexTokenizer output (first 20 tokens): {regex_tokens[:20]}")

def build_demo_corpus() -> List[str]:
    """ 
    Build a small corpus for CountVectorizer demonstration.
    @return (List[str]): List of documents (strings). 
    """
    return [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

def main() -> None:
    """ 
    Main entry point of the program.
    """
    # Instantiate tokenizers.
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()
    # Prepare test sentences and demo tokenizers.
    sentences = prepare_test_sentences()
    demo_tokenizers(simple_tokenizer, regex_tokenizer, sentences)
    # Demo CountVectorizer with RegexTokenizer.
    corpus = build_demo_corpus()
    demo_count_vectorizer(regex_tokenizer, corpus)
    # Demo tokenization on dataset sample (no path required; loader has fallback).
    demo_dataset_tokenization(simple_tokenizer, regex_tokenizer, dataset_path = None)

if __name__ == "__main__":
    main()