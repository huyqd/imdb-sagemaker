import argparse
import os
import warnings

import pandas as pd
import torch
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    parser.add_argument("--model_name", type=str)
    args, _ = parser.parse_known_args()

    split_ratio = args.train_test_split_ratio
    model_name = args.model_name

    print("Received arguments {}".format(args))

    raw_data = pd.read_csv(os.path.join("/opt/ml/processing/input", "raw.csv"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(raw_data['text'], raw_data['label'], test_size=split_ratio,
                                                        random_state=42)

    tokenized_X_train = tokenizer(X_train.tolist(), truncation=True, padding=True)
    ii_train = torch.Tensor(tokenized_X_train['input_ids']).long()
    am_train = torch.Tensor(tokenized_X_train['attention_mask']).long()
    y_train = torch.Tensor(y_train.tolist()).long()

    tokenized_X_test = tokenizer(X_test.tolist(), truncation=True, padding=True)
    ii_test = torch.Tensor(tokenized_X_test['input_ids']).long()
    am_test = torch.Tensor(tokenized_X_test['attention_mask']).long()
    y_test = torch.Tensor(y_test.tolist()).long()

    train_output_path = "/opt/ml/processing/train/"
    test_output_path = "/opt/ml/processing/test/"

    print(f"Saving training dateset to {train_output_path}")
    torch.save(ii_train, os.path.join(train_output_path, 'ii_train.pt'))
    torch.save(am_train, os.path.join(train_output_path, 'am_train.pt'))
    torch.save(y_train, os.path.join(train_output_path, 'y_train.pt'))

    print(f"Saving testing dateset to {test_output_path}")
    torch.save(ii_test, os.path.join(test_output_path, 'ii_test.pt'))
    torch.save(am_test, os.path.join(test_output_path, 'am_test.pt'))
    torch.save(y_test, os.path.join(test_output_path, 'y_test.pt'))
