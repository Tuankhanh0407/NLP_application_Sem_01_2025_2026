# Import necessary libraries.
from data_utils import HWUDataLoader
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from typing import List

class Word2VecAvgDenseTask:
    """ 
    Train Word2Vec embeddings, create average sentence vectors, and classify with a Dense network.

    Attributes:
    - w2v_model (Word2Vec): The trained Word2Vec model.
    - vector_size (int): Dimension of word embeddings.
    - num_classes (int): Number of intent classes. 
    """
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1) -> None:
        """ 
        Initialize the Word2VecAvgDenseTask class.
        @param vector_size (int): Embedding dimension for Word2Vec.
        @param window (int): Context window size for Word2Vec.
        @param min_count (int): Minimum word frequency for Word2Vec vocabulary.
        """
        self.w2v_model: Word2Vec = None
        self.vector_size: int = vector_size
        self.window: int = window
        self.min_count: int = min_count
        self.num_classes: int = 0

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        """ 
        Tokenize texts by simple whitespace split.
        @param texts (List[str]): A list of raw text strings.
        @return (List[List[str]]): List of token lists. 
        """
        return [t.strip().split() for t in texts]
    
    def train_w2v(self, sentences: List[List[str]]) -> None:
        """ 
        Train the Word2Vec model on tokenized sentences.
        @param sentences (List[List[str]]): List of tokenized sentences. 
        """
        self.w2v_model = Word2Vec(
            sentences = sentences,
            vector_size = self.vector_size,
            window = self.window,
            min_count = self.min_count,
            workers = 4
        )

    def sentence_to_avg_vector(self, tokens: List[str]) -> np.ndarray:
        """ 
        Convert a token list to an average embedding vector using the trained Word2Vec.
        @param tokens (List[str]): List of tokens for a sentence.
        @return (np.ndarray): Average embedding vector of shape (vector_size,). 
        """
        vecs = []
        for tok in tokens:
            if tok in self.w2v_model.wv:
                vecs.append(self.w2v_model.wv[tok])
        if len(vecs) == 0:
            return np.zeros((self.vector_size,), dtype = np.float32)
        return np.mean(np.stack(vecs, axis = 0), axis = 0)
    
    def build_dense_model(self) -> Sequential:
        """ 
        Build the Keras Dense classifier.
        @return (Sequential): Compiled Keras Sequential model. 
        """
        model = Sequential([
            Dense(128, activation = 'relu', input_shape = (self.vector_size,)),
            Dropout(0.5),
            Dense(self.num_classes, activation = 'softmax')
        ])
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        return model
    
    def train_and_evaluate(self) -> None:
        """ 
        Execute full pipeline: Load data, fit Word2Vec, create averaged vectors, train dense model, evaluate. 
        """
        loader = HWUDataLoader()
        df_train, df_val, df_test = loader.ensure_data()
        df_train, df_val, df_test, _ = loader.encode_labels(df_train, df_val, df_test)
        self.num_classes = int(max(df_train["label"].max(), df_val["label"].max(), df_test["label"].max())) + 1
        train_tokens = self.tokenize(df_train["text"].tolist())
        val_tokens = self.tokenize(df_val["text"].tolist())
        test_tokens = self.tokenize(df_test["text"].tolist())
        self.train_w2v(train_tokens)
        X_train = np.stack([self.sentence_to_avg_vector(toks) for toks in train_tokens], axis = 0)
        X_val = np.stack([self.sentence_to_avg_vector(toks) for toks in val_tokens], axis = 0)
        X_test = np.stack([self.sentence_to_avg_vector(toks) for toks in test_tokens], axis = 0)
        y_train = df_train["label"].values
        y_val = df_val["label"].values
        y_test = df_test["label"].values
        model = self.build_dense_model()
        es = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
        model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 50, batch_size = 32, callbacks = [es], verbose = 0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)
        y_pred = np.argmax(model.predict(X_test, verbose = 0), axis = 1)
        print("=== Word2Vec (Avg) and Dense results ===")
        print(classification_report(y_test, y_pred, digits = 4))
        f1 = f1_score(y_test, y_pred, average = "macro")
        print(f"Macro F1-score (test): {f1:.4f}")
        print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

def main() -> None:
    """ 
    Execute the task 2 demonstration: Word2Vec (Avg) and Dense task.
    """
    task = Word2VecAvgDenseTask()
    task.train_and_evaluate()

if __name__ == "__main__":
    main()