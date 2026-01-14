# Import necessary libraries.
from data_utils import HWUDataLoader
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore

class LSTMPretrainedTask:
    """ 
    Build an LSTM classifier using pre-trained embeddings from Word2Vec.

    Attributes:
    - tokenizer (Tokenizer): Keras tokenizer for building vocabulary and sequences.
    - max_len (int): Maximum sequence length for padding.
    - num_classes (int): Number of intent classes.
    - embedding_dim (int): Dimension of Word2Vec embeddings. 
    """
    def __init__(self, max_len: int = 50, embedding_dim: int = 100) -> None:
        """ 
        Initialize the LSTMPretrainedTask class.
        @param max_len (int): Maximum padding length.
        @param embedding_dim (int): Word2Vec embedding dimension. 
        """
        self.tokenizer: Tokenizer = Tokenizer(num_words = None, oov_token = "<UNK>")
        self.max_len: int = max_len
        self.num_classes: int = 0
        self.embedding_dim: int = embedding_dim

    def fit_tokenizer(self, texts: list) -> None:
        """ 
        Fit tokenizer on training texts.
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
    
    def train_w2v(self, sentences: list) -> Word2Vec:
        """ 
        Train a Word2Vec model on tokenized sentences for pre-trained embeddings.
        @param sentences (list): List of token lists.
        @return (Word2Vec): Trained Word2Vec model.
        """
        return Word2Vec(sentences = sentences, vector_size = self.embedding_dim, window = 5, min_count = 1, workers = 4)
    
    def build_embedding_matrix(self, w2v_model: Word2Vec) -> np.ndarray:
        """ 
        Build embedding matrix aligned with tokenizer word_index using trained Word2Vec.
        @param w2v_model (Word2Vec): Trained Word2Vec model.
        @return (np.ndarray): Embedding matrix of shape (vocab_size, embedding_dim). 
        """
        word_index = self.tokenizer.word_index
        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim), dtype = np.float32)
        for word, i, in word_index.items():
            if word in w2v_model.wv:
                embedding_matrix[i] = w2v_model.wv[word]
        return embedding_matrix
    
    def build_model(self, vocab_size: int, embedding_matrix: np.ndarray) -> Sequential:
        """ 
        Build LSTM model with frozen pre-trained embeddings.
        @param vocab_size (int): Vocabulary size.
        @param embedding_matrix (np.ndarray): Pre-trained embedding weights.
        @return (Sequential): Compiled Keras model. 
        """
        model = Sequential([
            Embedding(input_dim = vocab_size, output_dim = self.embedding_dim, weights = [embedding_matrix], input_length = self.max_len, trainable = False),
            LSTM(128, dropout = 0.2, recurrent_dropout = 0.2),
            Dense(self.num_classes, activation = 'softmax')
        ])
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        return model
    
    def train_and_evaluate(self) -> None:
        """ 
        Execute pipeline: Load data, tokenize/pad, pre-train embeddings, build LSTM, train, evaluate. 
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
        train_tokens = [t.split() for t in df_train["text"].tolist()]
        w2v_model = self.train_w2v(train_tokens)
        embedding_matrix = self.build_embedding_matrix(w2v_model)
        vocab_size = embedding_matrix.shape[0]
        model = self.build_model(vocab_size, embedding_matrix)
        es = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
        model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 50, batch_size = 32, callbacks = [es], verbose = 0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)
        y_pred = np.argmax(model.predict(X_test, verbose = 0), axis = 1)
        print("=== LSTM with pre-trained embeddings results ===")
        print(classification_report(y_test, y_pred, digits = 4))
        f1 = f1_score(y_test, y_pred, average = "macro")
        print(f"Macro F1-score (test): {f1:.4f}")
        print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

def main() -> None:
    """ 
    Execute the task 3 demonstration: LSTM with pre-trained embeddings task. 
    """
    task = LSTMPretrainedTask()
    task.train_and_evaluate()

if __name__ == "__main__":
    main()