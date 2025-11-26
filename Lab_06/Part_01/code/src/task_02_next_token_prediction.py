""" 
(i) Problem description:
Generate text continuation using next token prediction.

Input:
- Prompt sentence.

Output:
- Generated text continuation.

(ii) Approach:
- Use GPT-based model through Hugging Face pipeline.
- Leverage decoder-only architecture for autoregressive text generation.
"""

# Import necessary libraries
from transformers import pipeline

class TextGenerator:
    """ 
    Perform text generation using next token prediction. 
    """

    def __init__(self):
        """ 
        Initialize the text generation pipeline. 
        """
        self.generator = pipeline("text-generation", model = "distilgpt2")

    def generate_text(self, prompt, max_length = 50, num_return_sequences = 1):
        """ 
        Generate text continuation from prompt.
        @param prompt (str): Input prompt sentence.
        @param max_length (int): Maximum total length of generated text.
        @param num_return_sequences (int): Number of sequences to generate.
        @return (list): List of generated text sequences. 
        """
        generated_texts = self.generator(
            prompt,
            max_length = max_length,
            num_return_sequences = num_return_sequences,
            truncation = True
        )
        return generated_texts
    
    def display_generated_texts(self, prompt, generated_texts):
        """ 
        Display generated texts in a formatted way.
        @param prompt (str): Original prompt.
        @param generated_texts (list): List of generated text dictionaries. 
        """
        print(f"Prompt: '{prompt}'")
        print("Generated text:")
        for i, text in enumerate(generated_texts, 1):
            generated_text = text['generated_text'].replace('\n', ' ').strip()
            print(f"{i}. {generated_text}")

def run_next_token_prediction():
    """ 
    Execute next token prediction task. 
    """
    try:
        generator = TextGenerator()
        # Test prompt
        prompt = "The best thing about learning NLP is"
        # Generate text
        generated_texts = generator.generate_text(
            prompt,
            max_length = 50,
            num_return_sequences = 1
        )
        # Display results
        generator.display_generated_texts(prompt, generated_texts)
    except Exception as e:
        print(f"Error in next token prediction: {e}")