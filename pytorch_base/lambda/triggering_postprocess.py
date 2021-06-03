import json
import urllib

import boto3

s3 = boto3.client('s3')


def lambda_handler(event, context):
    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    fname = key.split('/')[-1].replace('.out', '')
    key = f"batch/input/{fname}"
    s3.delete_object(Bucket=bucket, Key=key)

    return {
        'statusCode': 200,
        'body': json.dumps(f"Successfully deleted {key}")
    }
