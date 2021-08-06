import io
import json

import pandas as pd
import tensorflow as tf

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


def model_fn(model_dir):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
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
    features = dict(tokenizer(input_data['text'].tolist(), truncation=True, padding="max_length", return_tensors='tf'))

    logits = model.predict(features).logits
    
    return logits
