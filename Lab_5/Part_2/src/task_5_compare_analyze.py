# Import necessary libraries.
from data_utils import HWUDataLoader
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from typing import Any, Dict, List, Tuple

class ComparisonRunner:
    """ 
    Train all four pipelines and compare macro F1-score and test loss.

    Attributes:
    - results (Dict[str, Dict[str, Any]]): Aggregated metrics for each pipeline including F1 and loss.
    - df_splits (Tuple): Cached train/val/test dataframes and labels for reuse.
    """
    def __init__(self) -> None:
        """ 
        Initialize the ComparisonRunner class. 
        """
        self.results: Dict[str, Dict[str, Any]] = {}
        self.df_splits = None

    def load_and_encode(self) -> Tuple[list, list, list, np.ndarray, np.ndarray, np.ndarray]:
        """ 
        Load HWU and encode labels.
        @return (Tuple[list, list, list, np.ndarray, np.ndarray, np.ndarray]): Raw text lists and label arrays for train/val/test. 
        """
        loader = HWUDataLoader()
        df_train, df_val, df_test = loader.ensure_data()
        df_train, df_val, df_test, _ = loader.encode_labels(df_train, df_val, df_test)
        return (
            df_train["text"].tolist(), df_val["text"].tolist(), df_test["text"].tolist(),
            df_train["label"].values, df_val["label"].values, df_test["label"].values
        )
    
    def run_tfidf_logreg(self) -> None:
        """ 
        Train TF-IDF and Logistic Regression and record macro F1 (no loss available). 
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_encode()
        pipeline = make_pipeline(TfidfVectorizer(max_features = 5000), LogisticRegression(max_iter = 1000))
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average = "macro")
        self.results["TF-IDF and Logistic Regression"] = {"f1_macro": float(f1), "test_loss": None}

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        """ 
        Tokenize texts by whitespace.
        @param texts (List[str]): Raw text list.
        @return (List[List[str]]): Token lists. 
        """
        return [t.strip().split() for t in texts]
    
    def run_w2v_avg_dense(self) -> None:
        """ 
        Train Word2Vec (Avg) and Dense and record macro F1 and test loss. 
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_encode()
        train_tokens = self.tokenize(X_train)
        val_tokens = self.tokenize(X_val)
        test_tokens = self.tokenize(X_test)
        w2v = Word2Vec(sentences = train_tokens, vector_size = 100, window = 5, min_count = 1, workers = 4)

        def avg_vec(tokens: List[str]) -> np.ndarray:
            vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
            if len(vecs) == 0:
                return np.zeros((100,), dtype = np.float32)
            return np.mean(np.stack(vecs, axis = 0), axis = 0)
        
        X_train_avg = np.stack([avg_vec(t) for t in train_tokens], axis = 0)
        X_val_avg = np.stack([avg_vec(t) for t in val_tokens], axis = 0)
        X_test_avg = np.stack([avg_vec(t) for t in test_tokens], axis = 0)
        num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
        model = Sequential([Dense(128, activation = 'relu', input_shape = (100,)), Dropout(0.5), Dense(num_classes, activation = 'softmax')])
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        es = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
        model.fit(X_train_avg, y_train, validation_data = (X_val_avg, y_val), epochs = 50, batch_size = 32, callbacks = [es], verbose = 0)
        test_loss, _ = model.evaluate(X_test_avg, y_test, verbose = 0)
        y_pred = np.argmax(model.predict(X_test_avg, verbose = 0), axis = 1)
        f1 = f1_score(y_test, y_pred, average = "macro")
        self.results["Word2Vec (Avg) and Dense"] = {"f1_macro": float(f1), "test_loss": float(test_loss)}

    def run_lstm_pretrained(self) -> None:
        """ 
        Train LSTM with pre-trained embeddings and record macro F1 and test loss. 
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_encode()
        tok = Tokenizer(num_words = None, oov_token = "<UNK>")
        tok.fit_on_texts(X_train)

        def to_pad(texts: List[str]) -> np.ndarray:
            return pad_sequences(tok.texts_to_sequences(texts), maxlen = 50, padding = 'post')
        
        X_train_pad = to_pad(X_train)
        X_val_pad = to_pad(X_val)
        X_test_pad = to_pad(X_test)
        train_tokens = [t.split() for t in X_train]
        w2v = Word2Vec(sentences = train_tokens, vector_size = 100, window = 5, min_count = 1, workers = 4)
        vocab_size = len(tok.word_index) + 1
        emb_matrix = np.zeros((vocab_size, 100), dtype = np.float32)
        for w, i in tok.word_index.items():
            if w in w2v.wv:
                emb_matrix[i] = w2v.wv[w]
        num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
        model = Sequential([Embedding(vocab_size, 100, input_length = 50, weights = [emb_matrix], trainable = False), LSTM(128, dropout = 0.2, recurrent_dropout = 0.2), Dense(num_classes, activation = 'softmax')])
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        es = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
        model.fit(X_train_pad, y_train, validation_data = (X_val_pad, y_val), epochs = 50, batch_size = 32, callbacks = [es], verbose = 0)
        test_loss, _ = model.evaluate(X_test_pad, y_test, verbose = 0)
        y_pred = np.argmax(model.predict(X_test_pad, verbose = 0), axis = 1)
        f1 = f1_score(y_test, y_pred, average = "macro")
        self.results["Embedding (pre-trained) and LSTM"] = {"f1_macro": float(f1), "test_loss": float(test_loss)}

    def run_lstm_scratch(self) -> None:
        """ 
        Train LSTM with trainable embeddings from scratch and record macro F1 and test loss.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_encode()
        tok = Tokenizer(num_words = None, oov_token = "<UNK>")
        tok.fit_on_texts(X_train)

        def to_pad(texts: List[str]) -> np.ndarray:
            return pad_sequences(tok.texts_to_sequences(texts), maxlen = 50, padding = 'post')
        
        X_train_pad = to_pad(X_train)
        X_val_pad = to_pad(X_val)
        X_test_pad = to_pad(X_test)
        vocab_size = len(tok.word_index) + 1
        num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
        model = Sequential([Embedding(vocab_size, 100, input_length = 50), LSTM(128, dropout = 0.2, recurrent_dropout = 0.2), Dense(num_classes, activation = 'softmax')])
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        es = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
        model.fit(X_train_pad, y_train, validation_data = (X_val_pad, y_val), epochs = 50, batch_size = 32, callbacks = [es], verbose = 0)
        test_loss, _ = model.evaluate(X_test_pad, y_test, verbose = 0)
        y_pred = np.argmax(model.predict(X_test_pad, verbose = 0), axis = 1)
        f1 = f1_score(y_test, y_pred, average = "macro")
        self.results["Embedding (scratch) + LSTM"] = {"f1_macro": float(f1), "test_loss": float(test_loss)}

    def print_table(self) -> None:
        """ 
        Print a simple comparison table of F1 macro and test loss. 
        """
        print("\n=== Comparison (macro F1 and test loss) ===")
        print(f"{'Pipeline':40s} | {'F1 (macro)':10s} | {'Test loss':10s}")
        print("-" * 68)
        for name, metrics in self.results.items():
            f1 = metrics['f1_macro']
            loss = metrics['test_loss']
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            print(f"{name:40s} | {f1:.4f}   | {loss_str}")

    def run_all(self) -> None:
        """ 
        Run all the four pipelines sequentially and print comparison. 
        """
        self.run_tfidf_logreg()
        self.run_w2v_avg_dense()
        self.run_lstm_pretrained()
        self.run_lstm_scratch()
        self.print_table()

def main() -> None:
    """ 
    Execute the task 5 demonstration: Run all pipelines and comparison. 
    """
    runner = ComparisonRunner()
    runner.run_all()

if __name__ == "__main__":
    main()