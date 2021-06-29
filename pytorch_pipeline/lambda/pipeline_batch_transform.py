import json
import os
import urllib
from time import gmtime, strftime

import boto3

s3 = boto3.client('s3')
sm = boto3.client('sagemaker')
role = 'arn:aws:iam::611215368770:role/service-role/AmazonSageMaker-ExecutionRole-20210517T164837'


def lambda_handler(event, context):
    model_package_group_name = f"Imdb-Reviews-1624825099"
    response = sm.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
        MaxResults=100,
    )
    approved_packages = response["ModelPackageSummaryList"]
    
    # Fetch more packages if none returned with continuation token
    while len(approved_packages) == 0 and "NextToken" in response:
        response = sm.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
            NextToken=response["NextToken"],
        )
        approved_packages.extend(response["ModelPackageSummaryList"])
    
    # Return error if no packages found
    if len(approved_packages) == 0:
        error_message = (
            f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
        )
        logger.error(error_message)
        raise Exception(error_message)
    
    # Return the pmodel package arn
    model_package_arn = approved_packages[0]["ModelPackageArn"]
    
    all_model_names = [model['ModelName'] for model in sm.list_models()['Models']]
    model_name = 'DEMO-modelregistry-model-imdb'
    print("Model name : {}".format(model_name))
    if model_name not in all_model_names:
        primary_container = {
            'ModelPackageName': model_package_arn,
        }
        
        create_model_respose = sm.create_model(
            ModelName = model_name,
            ExecutionRoleArn = role,
            PrimaryContainer = primary_container
        )
        print("Model arn : {}".format(create_model_respose["ModelArn"]))
        
    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    input_location = f"s3://{bucket}/{key}"

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

    sm.create_transform_job(**request)
    print("Created Transform job with name: ", batch_job_name)

    return {
        'statusCode': 200,
        'body': json.dumps(f'Successfully created Transform job with name: {batch_job_name}')
    }
