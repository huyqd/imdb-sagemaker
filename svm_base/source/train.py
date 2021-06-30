import argparse
import logging
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, DistilBertModel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import joblib


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load model and tokenizer
    model_name = 'distilbert-base-uncased'
    model = DistilBertModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))

    # Tokenize
    train_texts, train_labels = train_data['text'].tolist(), train_data['label'].tolist()
    tokenized_train = tokenizer(train_texts, truncation=True, padding=True)

    input_ids = torch.Tensor(tokenized_train['input_ids']).long()
    attention_masks = torch.Tensor(tokenized_train['attention_mask']).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids.to(device)
    attention_masks.to(device)
    

    model.eval()
    with torch.no_grad():
        res = model(input_ids=input_ids, attention_mask=attention_masks)
    X_train = res['last_hidden_state'][:, 0].numpy()
    
    model = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    model.fit(X_train, train_labels)
    
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))