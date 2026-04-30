import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ROOT = os.path.expanduser("~/har-trustworthy-ml/data/UCI_HAR_Dataset copy/")

LABELS = {1: "Walking", 2: "Walking Upstairs", 3: "Walking Downstairs",
          4: "Sitting", 5: "Standing", 6: "Laying"}

def load_data(split: str, base_path: str = DATA_ROOT) -> tuple:
    try:
        X = pd.read_csv(
            os.path.join(base_path, split, f"X_{split}.txt"),
            delim_whitespace=True, header=None
        )
        y = pd.read_csv(
            os.path.join(base_path, split, f"y_{split}.txt"),
            header=None
        ).squeeze()
        return X, y
    except Exception as e:
        print(f"Error loading {split} data: {e}")
        return None, None

def get_splits():
    X_train, y_train = load_data("train")
    X_test,  y_test  = load_data("test")
    if X_train is None or X_test is None:
        return None, None, None, None, None, None
        
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_test, y_test, test_size=0.70, random_state=42, stratify=y_test
    )
    return X_train, y_train, X_cal, y_cal, X_eval, y_eval
