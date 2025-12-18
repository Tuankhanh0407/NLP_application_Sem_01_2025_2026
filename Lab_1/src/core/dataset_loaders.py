# Import necessary libraries.
import os
from typing import Optional

def load_raw_text_data(dataset_path: Optional[str] = None) -> str:
    """ 
    Load raw text data from a dataset path. If the path is not available, return a built-in sample text so the rest of the labs can run without manual dataset setup.
    @param dataset_path (Optional[str]): Path to dataset directory or file.
    @return (str): Raw text content (string). 
    """
    # If a path is provided and exists, try to read files inside it.
    if dataset_path:
        # If it is a file, try to read it.
        if os.path.isfile(dataset_path):
            try:
                with open(dataset_path, 'r', encoding = 'utf-8') as f:
                    return f.read()
            except Exception:
                pass

        # If it is a directory, try to read all .txt files and concatenate.
        if os.path.isdir(dataset_path):
            collected = []
            for root, _, files in os.walk(dataset_path):
                for file_name in files:
                    if file_name.lower().endswith('.txt') or file_name.lower().endswith('.conllu'):
                        try:
                            with open(os.path.join(root, file_name), 'r', encoding = 'utf-8') as f:
                                collected.append(f.read())
                        except Exception:
                            continue
            if collected:
                return "\n".join(collected)
    # Fallback built-in sample text (sufficient for demonstration).
    sample_text = (
        "Hello, world! This is a test. NLP is fascinating ... isn't it? "
        "Let's see how it handles 123 numbers and punctuation! "
        "The quick brown fox jumps over the lazy dog. "
        "In 2025, natural language processing continues to evolve."
    )
    return sample_text