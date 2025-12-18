# Import necessary libraries.
import spacy
from typing import List, Optional

def ensure_model(model_name: str = "en_core_web_md") -> spacy.language.Language:
    """ 
    Ensure the specified spaCy model is available. If not, download it programmatically.
    @param model_name (str): The name of the spaCy model to load.
    @return (spacy.language.Language): Loaded spaCy language pipeline. 
    """
    try:
        # Try to import model package (fast path).
        return spacy.load(model_name)
    except Exception:
        # If loading fails, attempt to download the model programmatically.
        print(f"[Model] spaCy model '{model_name}' not found. Attempting to download...")
        try:
            # Use spaCy CLI to download the model.
            import spacy.cli
            spacy.cli.download(model_name)
            return spacy.load(model_name)
        except Exception as e:
            print("[Model] Failed to download or load the spaCy model.")
            raise e
        
def find_main_verb(doc: spacy.tokens.Doc) -> Optional[spacy.tokens.Token]:
    """ 
    Find the main verb (token with dependency label ROOT) in a parsed Doc.
    @param doc (spacy.tokens.Doc): The parsed spaCy Doc object.
    @return (Optional[spacy.tokens.Token]): The token that is the ROOT (main verb) or None if not found. 
    """
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None

def noun_chunks_custom(doc: spacy.tokens.Doc) -> List[List[spacy.tokens.Token]]:
    """ 
    A simple custom noun chunk extractor: For each NOUN, collect its determiners, compounds, and amod children.
    @param doc (spacy.tokens.Doc): The parsed spaCy Doc object.
    @return (List[List.spacy.tokens.Token]): List of noun chunk token lists. 
    """
    chunks = []
    for token in doc:
        if token.pos_ == "NOUN":
            chunk_tokens = []
            # Include left-side compounds and determiners by scanning token's lefts.
            for left in token.lefts:
                if left.dep_ in {"det", "amod", "compound", "nummod", "poss"}:
                    chunk_tokens.append(left)
            # Include the noun itself.
            chunk_tokens.append(token)
            # Include right-side modifiers that are part of the noun phrase (in example, amod on right).
            for right in token.rights:
                if right.dep_ in {"amod", "compound", "nummod"}:
                    chunk_tokens.append(right)
            # Sort tokens by position.
            chunk_tokens_sorted = sorted(set(chunk_tokens), key = lambda t: t.i)
            chunks.append(chunk_tokens_sorted)
    return chunks

def get_path_to_root(token: spacy.tokens.Token) -> List[spacy.tokens.Token]:
    """ 
    Return the path from a token up to the root (inclusive).
    @param token (spacy.tokens.Token): The starting token.
    @return (List[spacy.tokens.Token]): List of tokens from the starting token up to the root. 
    """
    path = [token]
    current = token
    while current.dep_ != "ROOT" and current.head != current:
        current = current.head
        path.append(current)
    return path