# Import necessary libraries.
import csv
import io
import os
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple

class HWUDataLoader:
    """ 
    Fetch, prepare, and split the HWU intent dataset or provide a synthetic fallback.

    Attributes:
    - root_dir (str): The root directory where downloaded or generated data files are stored.
    - filenames (Dict[str, str]): Mapping of split names to expected CSV filenames. 
    """
    def __init__(self, root_dir: str = "data") -> None:
        """ 
        Initialize the HWUDataLoader class.
        @param root_dir (str): Target folder for data files. 
        """
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok = True)
        self.filenames: Dict[str, str] = {
            "train": os.path.join(self.root_dir, "hwu_train.csv"),
            "val": os.path.join(self.root_dir, "hwu_val.csv"),
            "test": os.path.join(self.root_dir, "hwu_test.csv"),
        }

    def download_hwu_zip(self) -> str:
        """ 
        Attempt to download HWU dataset ZIP from a public mirror.
        @return (str): Path to the downloaded ZIP file or an empty string if failed. 
        """
        # Public mirror example (may change over time). If failing, caller will fallback to synthetic data.
        url_candidates = [
            # These URLs are examples; if none work, fallback is used.
            "https://raw.githubusercontent.com/LightTagTeam/datasets/master/hwu/hwu_train.csv",
            "https://raw.githubusercontent.com/LightTagTeam/datasets/master/hwu/hwu_val.csv",
            "https://raw.githubusercontent.com/LightTagTeam/datasets/master/hwu/hwu_test.csv",
        ]
        try:
            # Try direct CSV endpoints first.
            for split, url in zip(["train", "val", "test"], url_candidates):
                out_path = self.filenames[split]
                if not os.path.exists(out_path):
                    resp = requests.get(url, timeout = 30)
                    resp.raise_for_status()
                    with io.open(out_path, "w", encoding = "utf-8", newline = "") as f:
                        f.write(resp.text)
            return "CSV_OK"
        except Exception:
            return ""
        
    def generate_synthetic_hwu(self) -> None:
        """ 
        Generate a small synthetic HWU-like dataset with text-intent pairs. 
        """
        synth_data = {
            "train": [
                ("what is the weather like today", "weather_query"),
                ("set a reminder to call mom", "reminder_create"),
                ("book a flight from hanoi to da nang", "flight_search"),
                ("turn on the living room lights", "smart_home"),
                ("play some relaxing music", "music_play"),
                ("remind me to not forget my keys", "reminder_create"),
                ("find a flight to london", "flight_search"),
                ("is it rainy or sunny tomorrow", "weather_query"),
            ],
            "val": [
                ("can you remind me to call dad", "reminder_create"),
                ("find flights from new york to london", "flight_search"),
            ],
            "test": [
                ("can you remind me to not call my mom", "reminder_create"),
                ("is it going to be sunny or rainy tomorrow", "weather_query"),
                ("find a flight from new york to london but not through paris", "flight_search"),
            ],
        }
        for split, rows in synth_data.items():
            with io.open(self.filenames[split], "w", encoding = "utf-8", newline = "") as f:
                writer = csv.writer(f, delimiter = "\t")
                for text, intent in rows:
                    writer.writerow([text, intent])

    def ensure_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ 
        Ensure HWU CSVs exist by downloading from a mirror or creating synthetic data, then load into dataframes.
        @return (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): DataFrames for train, val, and test splits with columns ["text", "intent"].
        """
        ok = self.download_hwu_zip()
        if not ok:
            self.generate_synthetic_hwu()
        df_train = pd.read_csv(self.filenames["train"], sep = '\t', header = None, names = ["text", "intent"])
        df_val = pd.read_csv(self.filenames['val'], sep = "\t", header = None, names = ["text", "intent"])
        df_test = pd.read_csv(self.filenames["test"], sep = "\t", header = None, names = ["text", "intent"])
        return df_train, df_val, df_test
    
    def encode_labels(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
        """ 
        Fit a LabelEncoder on combined intents and transform splits.
        @param df_train (pd.DataFrame): Training dataframe with ["text", "intent"].
        @param df_val (pd.DataFrame): Validation dataframe.
        @param df_test (pd.DataFrame): Test dataframe.
        @return (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]): Transformed dataframes with an added "label" int column and the fitted LabelEncoder.
        """
        le = LabelEncoder()
        all_intents = pd.concat([df_train["intent"], df_val["intent"], df_test["intent"]], axis = 0)
        le.fit(all_intents.values)
        for df in (df_train, df_val, df_test):
            df["label"] = le.transform(df["intent"].values)
        return df_train, df_val, df_test, le