import json
import os
import urllib
from time import gmtime, strftime

import boto3

s3 = boto3.client('s3')
model_name = os.environ['model_name']


def lambda_handler(event, context):
    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    input_location = f"s3://{bucket}/{key}"

    sagemaker = boto3.client('sagemaker')
    batch_job_name = "Batch-Transform-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    output_location = f"s3://{bucket}/batch/output/"

    request = {
        "TransformJobName": batch_job_name,
        "ModelName": model_name,
        "BatchStrategy": "SingleRecord",
        "TransformOutput": {
            "S3OutputPath": output_location,
            "Accept": "text/csv",
            "AssembleWith": "Line",
        },
        "TransformInput": {
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_location}},
            "ContentType": "text/csv",
            "SplitType": "Line",
            "CompressionType": "None",
        },
        "TransformResources": {"InstanceType": "ml.m5.large", "InstanceCount": 1},
    }

    sagemaker.create_transform_job(**request)
    print("Created Transform job with name: ", batch_job_name)

    return {
        'statusCode': 200,
        'body': json.dumps(f'Successfully created Transform job with name: {batch_job_name}')
    }
