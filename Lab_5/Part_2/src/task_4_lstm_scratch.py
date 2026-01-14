# Import necessary libraries.
from data_utils import HWUDataLoader
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore

class LSTMScratchTask:
    """ 
    Build an LSTM classifier with embeddings learned end-to-end from scratch.

    Attributes:
    - tokenizer (Tokenizer): Keras tokenizer for building vocabulary and sequences.
    - max_len (int): Maximum sequence length for padding.
    - num_classes (int): Number of intent classes.
    - embedding_dim (int): Dimension of the trainable embedding layer.
    """
    def __init__(self, max_len: int = 50, embedding_dim: int = 100) -> None:
        """ 
        Initialize the LSTMScratchTask class.
        @param max_len (int): Maximum padding length.
        @param embedding_dim (int): Embedding dimension to learn from scratch. 
        """
        self.tokenizer: Tokenizer = Tokenizer(num_words = None, oov_token = "<UNK>")
        self.max_len: int = max_len
        self.num_classes: int = 0
        self.embedding_dim: int = embedding_dim

    def fit_tokenizer(self, texts: list) -> None:
        """ 
        Fit tokenizer on training texts
        @param texts (list): List of raw training sentences. 
        """
        self.tokenizer.fit_on_texts(texts)

    def to_padded(self, texts: list) -> np.ndarray:
        """ 
        Convert texts to padded integer sequences.
        @param texts (list): List of raw text samples.
        @return (np.ndarray): 2D array of shape (n_samples, max_len) with padded indices. 
        """
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen = self.max_len, padding = 'post')
    
    def build_model(self, vocab_size: int) -> Sequential:
        """ 
        Build LSTM model with trainable embedding from scratch.
        @param vocab_size (int): Vocabulary size for the embedding layer.
        @return (Sequential): Compiled Keras model. 
        """
        model = Sequential([
            Embedding(input_dim = vocab_size, output_dim = self.embedding_dim, input_length = self.max_len),
            LSTM(128, dropout = 0.2, recurrent_dropout = 0.2),
            Dense(self.num_classes, activation = 'softmax')
        ])
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        return model
    
    def train_and_evaluate(self) -> None:
        """ 
        Execute pipeline: Load data, tokenize/pad, build LSTM from scratch, train, evaluate. 
        """
        loader = HWUDataLoader()
        df_train, df_val, df_test = loader.ensure_data()
        df_train, df_val, df_test, _ = loader.encode_labels(df_train, df_val, df_test)
        self.num_classes = int(max(df_train["label"].max(), df_val["label"].max(), df_test["label"].max())) + 1
        self.fit_tokenizer(df_train["text"].tolist())
        X_train = self.to_padded(df_train["text"].tolist())
        X_val = self.to_padded(df_val["text"].tolist())
        X_test = self.to_padded(df_test["text"].tolist())
        y_train = df_train["label"].values
        y_val = df_val["label"].values
        y_test = df_test["label"].values
        vocab_size = len(self.tokenizer.word_index) + 1
        model = self.build_model(vocab_size)
        es = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
        model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 50, batch_size = 32, callbacks = [es], verbose = 0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)
        y_pred = np.argmax(model.predict(X_test, verbose = 0), axis = 1)
        print("=== LSTM with embeddings learned from scratch results ===")
        print(classification_report(y_test, y_pred, digits = 4))
        f1 = f1_score(y_test, y_pred, average = "macro")
        print(f"Macro F1-score (test): {f1:.4f}")
        print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

def main() -> None:
    """ 
    Execute the task 4 demonstration: LSTM-from-scratch task. 
    """
    task = LSTMScratchTask()
    task.train_and_evaluate()

if __name__ == "__main__":
    main()