import json
import os
import tarfile
import pandas as pd

import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, recall_score, precision_score, confusion_matrix, roc_curve
from transformers import AutoModelForSequenceClassification

if __name__ == "__main__":
    model_path = "/opt/ml/processing/model"
    test_input_path = "/opt/ml/processing/test/"

    with tarfile.open(os.path.join(model_path, "model.tar.gz")) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=".")
    print("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(".")

    print("Loading test input data")
    ii_test = torch.load(os.path.join(test_input_path, 'ii_test.pt'))
    am_test = torch.load(os.path.join(test_input_path, 'am_test.pt'))
    y_test = torch.load(os.path.join(test_input_path, 'y_test.pt'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ii_test.to(device)
    am_test.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ii_test, attention_mask=am_test).logits
        sm = torch.softmax(logits, dim=1).detach().cpu().numpy()
        sm = sm[:, 1]
        predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()

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
                "false_positive_rates": fpr.tolist(),
                "true_positive_rates": tpr.tolist(),
            },
            "precision_recall_curve": {
                "precisions": precision.tolist(),
                "recalls": recall.tolist(),
            },
        }
    }

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
