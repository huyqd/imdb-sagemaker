import json
import os
import tarfile
import pandas as pd
import numpy as np
import joblib

import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, recall_score, \
    precision_score, confusion_matrix, roc_curve
from transformers import DistilBertModel

if __name__ == "__main__":
    model_path = "/opt/ml/processing/model"
    test_input_path = "/opt/ml/processing/test/"

    with tarfile.open(os.path.join(model_path, "model.tar.gz")) as tar:
        tar.extractall(path=".")
    print("Loading model")
    model = joblib.load('model.joblib')

    print("Loading test input data")
    ii_test = torch.load(os.path.join(test_input_path, 'ii_test.pt'))
    am_test = torch.load(os.path.join(test_input_path, 'am_test.pt'))
    y_test = torch.load(os.path.join(test_input_path, 'y_test.pt'))

    # Load model and tokenizer
    model_name = 'distilbert-base-uncased'
    model = DistilBertModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ii_test.to(device)
    am_test.to(device)

    model.eval()
    with torch.no_grad():
        res = model(input_ids=ii_test, attention_mask=am_test)
    X_test = res['last_hidden_state'][:, 0].numpy()

    logits = model.predict_proba(X_test)
    sm = np.exp(logits) / np.sum(np.exp(logits))
    predictions = model.predict(X_test)

    print("Creating classification evaluation report")
    fpr, tpr, thresholds = roc_curve(y_test, sm)
    precision, recall, thresholds = precision_recall_curve(y_test, sm)

    report_dict = {
        "binary_classification_metrics": {
            "confusion_matrix": pd.DataFrame(
                confusion_matrix(y_test, predictions), index=["0", "1"], columns=["0", "1"]
            ).to_dict(),
            "accuracy": {
                "value": accuracy_score(y_test, predictions),
                "standard_deviation": "NaN",
            },
            "roc_auc": {
                "value": roc_auc_score(y_test, predictions),
                "standard_deviation": "NaN",
            },
            "f1": {
                "value": f1_score(y_test, predictions),
                "standard_deviation": "NaN",
            },
            "recall": {
                "value": recall_score(y_test, predictions),
                "standard_deviation": "NaN",
            },
            "precision": {
                "value": precision_score(y_test, predictions),
                "standard_deviation": "NaN",
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": fpr,
                "true_positive_rates": tpr,
            },
            "precision_recall_curve": {
                "precisions": precision,
                "recalls": recall,
            },
        }
    }

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
