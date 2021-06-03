import argparse
import os
import warnings

import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    raw_data = pd.read_csv(os.path.join("/opt/ml/processing/input", "raw.csv"))
    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    train, test = train_test_split(raw_data, test_size=split_ratio, random_state=42)
    train["text"] = train["text"].apply(lambda x: x.encode("utf-8"))
    test["text"] = test["text"].apply(lambda x: x.encode("utf-8"))

    train_output_path = "/opt/ml/processing/train/train.csv"
    test_output_path = "/opt/ml/processing/test/test.csv"

    print(f"Saving training dateset to {train_output_path}")
    train.to_csv(train_output_path, index=False)

    print(f"Saving testing dateset to {test_output_path}")
    test.to_csv(test_output_path, index=False)
