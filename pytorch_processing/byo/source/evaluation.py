import json
import os
import tarfile

import torch
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
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
        predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()

    print("Creating classification evaluation report")
    report_dict = classification_report(y_test, predictions, output_dict=True)
    report_dict["accuracy"] = accuracy_score(y_test, predictions)
    report_dict["roc_auc"] = roc_auc_score(y_test, predictions)

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
