import argparse
import io
import json
import logging
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoConfig


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def model_fn(model_dir):
    config = AutoConfig.from_pretrained(os.path.join(model_dir, 'config.json'))
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_dir, 'pytorch_model.bin'),
                                                               config=config)

    return model


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        print("================ input sentences ===============")
        print(data)

        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            pass
        else:
            raise ValueError("Unsupported input type. Input type can be a string or an non-empty list. \
                             I got {}".format(data))

    elif request_content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body))
        data = df['text'].tolist()

    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

    return data


def predict_fn(input_data, model):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized_input = tokenizer(input_data, truncation=True, padding=True)
    model.eval()

    with torch.no_grad():
        y = model(**tokenized_input)[0]
        print("=============== inference result =================")
        print(y)
    return y
