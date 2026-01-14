# Import necessary libraries.
from data_utils import HWUDataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from typing import Tuple

class TFIDFLogRegTask:
    """ 
    Pipeline for TF-IDF and Logistic Regression baseline on HWU intents.

    Attributes:
    - pipeline (sklearn.pipeline.Pipeline): The TF-IDF vectorizer and Logistic Regression pipeline. 
    """
    def __init__(self) -> None:
        """ 
        Initialize the TFIDFLogRegTask class. 
        """
        self.pipeline = make_pipeline(
            TfidfVectorizer(max_features = 5000),
            LogisticRegression(max_iter = 1000)
        )

    def load_data(self) -> Tuple[list, list, list, np.ndarray, np.ndarray, np.ndarray]:
        """ 
        Load HWU data and encode labels.
        @return (Tuple[list, list, list, np.ndarray, np.ndarray, np.ndarray]): Text lists for train/val/test and label arrays y_train/y_val/y_test. 
        """
        loader = HWUDataLoader()
        df_train, df_val, df_test = loader.ensure_data()
        df_train, df_val, df_test, _ = loader.encode_labels(df_train, df_val, df_test)
        return (
            df_train["text"].tolist(),
            df_val["text"].tolist(),
            df_test["text"].tolist(),
            df_train["label"].values,
            df_val["label"].values,
            df_test["label"].values
        )
    
    def train_and_evaluate(self) -> None:
        """ 
        Fit the pipeline and print classification report and macro F1 on test set. 
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        print("=== TF-IDF and Logistic Regression results ===")
        print(classification_report(y_test, y_pred, digits = 4))
        f1 = f1_score(y_test, y_pred, average = "macro")
        print(f"Macro F1-score (test): {f1:.4f}")

def main() -> None:
    """ 
    Execute the task 1 demonstration: TF-IDF and Logistic Regression task.
    """
    task = TFIDFLogRegTask()
    task.train_and_evaluate()

if __name__ == "__main__":
    main()