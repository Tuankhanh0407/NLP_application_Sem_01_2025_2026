""" 
(i) Problem description:
Convert sentences to fixed-dimensional vector representations.

Input:
- Sentence text.

Output:
- Sentence embedding vector.

(ii) Approach:
- Use BERT model with mean pooling strategy.
- Apply attention mask to ignore padding tokens.
- Generate dense vector representations capturing semantic meaning.
"""

# Import necessary libraries
import torch
from transformers import AutoModel, AutoTokenizer

class SentenceEmbedder:
    """ 
    Generate sentence embeddings using BERT with mean pooling. 
    """

    def __init__(self, model_name = "distilbert-base-uncased"):
        """ 
        Initialize tokenizer and model.
        @param model_name (str): Name of pre-trained model to use. 
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Set model to evaluation mode
        self.model.eval()

    def tokenize_sentences(self, sentences):
        """ 
        Tokenize input sentences.
        @param sentences (list): List of sentences of tokenize.
        @return (dict): Tokenized inputs with attention masks. 
        """
        inputs = self.tokenizer(
            sentences,
            padding = True,
            truncation = True,
            return_tensors = 'pt',
            max_length = 512
        )
        return inputs
    
    def compute_embeddings(self, inputs):
        """ 
        Compute sentence embeddings using mean pooling.
        @param inputs (dict): Tokenized inputs.
        @return (torch.Tensor): Sentence embeddings. 
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        # Perfrom mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min = 1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        return sentence_embeddings
    
    def display_embedding_info(self, sentence, embedding):
        """ 
        Display embedding information.
        @param sentence (str): Original sentence.
        @param embedding (torch.Tensor): Computed embedding vector. 
        """
        print(f"Sentence: '{sentence}'")
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 10 dimensions: {embedding[0][:10]}")
        print(f"Embedding norm: {torch.norm(embedding):.4f}")

def run_sentence_representation():
    """ 
    Execute sentence representation task. 
    """
    try:
        embedder = SentenceEmbedder()
        # Test sentence
        sentences = ["This is a sample sentence."]
        # Tokenize sentences
        inputs = embedder.tokenize_sentences(sentences)
        # Compute embeddings
        sentence_embeddings = embedder.compute_embeddings(inputs)
        # Display results
        embedder.display_embedding_info(sentences[0], sentence_embeddings)
    except Exception as e:
        print(f"Error in sentence representation: {e}")