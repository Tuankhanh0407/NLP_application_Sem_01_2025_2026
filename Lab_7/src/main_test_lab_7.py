""" 
(i) Problem description:
This program demonstrates dependency parsing tasks using spaCy.
It implements multiple sub-tasks from the provided lab:
- Load spaCy model (en_core_web_md), downloading it if necessary.
- Parse example sentences.
- Visualize dependency trees (save HTML).
- Print token attributes and children.
- Extract (subject, verb, object) triplets.
- Extract adjectives modifying nouns.
- Provide utility functions: Find main verb, noun-chunk extraction, path to root.

Input:
- No external input required. The program uses built-in example sentences from the lab.

Output:
- Console output showing parsing results and extracted information.
- HTML files saved for dependency visualizations.

(ii) Approach:
- Use spaCy to parse sentences and access token attributes.
- Implement modular functions for each sub-task.
- Provide robust checks for model availability and download if missing.
"""

# Import necessary libraries.
from parser_exercises import ensure_model, find_main_verb, get_path_to_root, noun_chunks_custom
from parser_extractors import extract_adjectives, extract_triplets
from parser_token_info import print_token_info
from parser_visualize import visualize_sentence
import pathlib

def run_all_examples(nlp):
    """ 
    Run all example tasks using the provided spaCy NLP pipeline.
    @param nlp (spacy.lang): The loaded spaCy language pipeline. 
    """
    # Visualization example and questions from lab.
    text_1 = "The quick brown fox jumps over the lazy dog."
    visualize_sentence(nlp, text_1, output_html_path = pathlib.Path("dep_visualization_fox.html"))
    # Print token info for example sentence from lab.
    text_2 = "Apple is looking at buying U.K. startup for $1 billion"
    print_token_info(nlp, text_2)
    # Extract triplets example.
    text_3 = "The cat chased the mouse and the dog watched them."
    extract_triplets(nlp, text_3)
    # Extract adjectives example.
    text_4 = "The big, fluffy white cat is sleeping on the warm mat."
    extract_adjectives(nlp, text_4)
    # Exercises.
    doc_1 = nlp("The quick brown fox jumps over the lazy dog.")
    main_verb = find_main_verb(doc_1)
    print(f"\n[Exercises] Main verb (ROOT) of sentence: {main_verb.text if main_verb else 'None'}")
    # Noun chunks custom.
    doc_2 = nlp("The big, fluffy white cat is sleeping on the warm mat.")
    chunks = noun_chunks_custom(doc_2)
    print("\n[Exercise] Custom noun chunks found:")
    for c in chunks:
        print(" -", " ".join([t.text for t in c]))
    # Path to root example.
    token = doc_1[3] # 'fox' in the first sentence.
    path = get_path_to_root(token)
    print("\n[Exercise] Path from token to root:")
    print(" -> ".join([t.text for t in path]))

def main():
    """ 
    Main entry point of the program. 
    """
    nlp = ensure_model() # Ensure model is available and loaded.
    run_all_examples(nlp)

if __name__ == "__main__":
    main()