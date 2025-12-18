# Import necessary libraries.
import pathlib
import spacy
from spacy import displacy
from typing import Optional

def visualize_sentence(nlp: spacy.language.Language, text: str, output_html_path: Optional[pathlib.Path] = None) -> None:
    """ 
    Render dependency visualization for a sentence and optionally save as HTML.
    @param nlp (spacy.language.Language): The loaded spaCy pipeline.
    @param text (str): The sentence to parse and visualize.
    @param output_html_path (Optional[pathlib.Path]): Path to save the HTML visualization. If none, file is not saved. 
    """
    doc = nlp(text)
    # Render HTML string for dependency visualization.
    html = displacy.render(doc, style = "dep", jupyter = False)
    if output_html_path:
        output_html_path.write_text(html, encoding = "utf-8")
        print(f"[Visualization] Dependency visualization saved to: {output_html_path.resolve()}")
    else:
        print("[Visualization] HTML not saved (no path provided).")
    # Also print a short summary to console.
    print(f"[Visualization] Sentence parsed: \"{text}\"")
    # Print root token.
    root = [token for token in doc if token.dep_ == "ROOT"]
    if root:
        print(f"[Visualization] ROOT token: {root[0].text}")