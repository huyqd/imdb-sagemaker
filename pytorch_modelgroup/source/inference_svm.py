import io
import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, DistilBertModel
import joblib


def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))

    return tokenizer, encoder, model


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
    tokenizer, encoder, model = model
    tokenized_input = tokenizer(input_data, truncation=True, padding=True)

    input_ids = torch.Tensor(tokenized_input['input_ids']).long()
    attention_masks = torch.Tensor(tokenized_input['attention_mask']).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    input_ids.to(device)
    attention_masks.to(device)

    encoder.eval()
    with torch.no_grad():
        res = encoder(input_ids=input_ids, attention_mask=attention_masks)
        
    encoded_input = res['last_hidden_state'][:, 0].numpy()
    
    logits = model.predict_proba(encoded_input)
        
    return logits
