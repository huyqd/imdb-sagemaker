import argparse
import logging
import os
import sys

import joblib
import torch
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from transformers import DistilBertModel

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

    # Load dataset
    ii_train = torch.load(os.path.join(args.train, 'ii_train.pt'))
    am_train = torch.load(os.path.join(args.train, 'am_train.pt'))
    train_labels = torch.load(os.path.join(args.train, 'y_train.pt'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ii_train.to(device)
    am_train.to(device)

    model.eval()
    with torch.no_grad():
        res = model(input_ids=ii_train, attention_mask=am_train)
    X_train = res['last_hidden_state'][:, 0].numpy()

    model = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    model.fit(X_train, train_labels)

    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
