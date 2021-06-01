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


def model_fn(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            pass
        else:
            raise ValueError("Unsupported input type. Input type can be a string or an non-empty list. \
                             I got {}".format(data))

    elif request_content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None)
        print(df.columns)
        df.columns = ['label', 'text']
        data = df['text'].tolist()

    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

    return data


def predict_fn(input_data, model):
    model, tokenizer = model
    tokenized_input = tokenizer(input_data, truncation=True, padding=True)
    
    input_ids = torch.Tensor(tokenized_input['input_ids']).long()
    attention_masks = torch.Tensor(tokenized_input['attention_mask']).long()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids.to(device)
    attention_masks.to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_masks).logits
    return logits
