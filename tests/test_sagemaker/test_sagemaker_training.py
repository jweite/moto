# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import boto3
import datetime
import pytz
import sure  # noqa
import time

from moto import mock_sagemaker
from moto.sts.models import ACCOUNT_ID
from nose.tools import assert_true, assert_equal, assert_raises, assert_regexp_matches

FAKE_ROLE_ARN = "arn:aws:iam::{}:role/FakeRole".format(ACCOUNT_ID)
TEST_REGION_NAME = "us-east-1"


@mock_sagemaker
def test_create_training_job():
    sagemaker = boto3.client("sagemaker", region_name=TEST_REGION_NAME)

    training_job_name = "MyTrainingJob"
    container = "382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:1"
    bucket = "my-bucket"
    prefix = "sagemaker/DEMO-breast-cancer-prediction/"

    params = {
        "RoleArn": FAKE_ROLE_ARN,
        "TrainingJobName": training_job_name,
        "AlgorithmSpecification": {
            "TrainingImage": container,
            "TrainingInputMode": "File",
        },
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.c4.2xlarge",
            "VolumeSizeInGB": 10,
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://{}/{}/train/".format(bucket, prefix),
                        "S3DataDistributionType": "ShardedByS3Key",
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None",
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://{}/{}/validation/".format(bucket, prefix),
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None",
            },
        ],
        "OutputDataConfig": {"S3OutputPath": "s3://{}/{}/".format(bucket, prefix)},
        "HyperParameters": {
            "feature_dim": "30",
            "mini_batch_size": "100",
            "predictor_type": "regressor",
            "epochs": "10",
            "num_models": "32",
            "loss": "absolute_loss",
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 60 * 60},
    }

    resp = sagemaker.create_training_job(**params)
    resp["TrainingJobArn"].should.match(
        r"^arn:aws:sagemaker:.*:.*:training-job/{}$".format(training_job_name)
    )

    resp = sagemaker.describe_training_job(TrainingJobName=training_job_name)
    resp["TrainingJobName"].should.equal(training_job_name)
    resp["TrainingJobArn"].should.match(
        r"^arn:aws:sagemaker:.*:.*:training-job/{}$".format(training_job_name)
    )
    assert_true(
        resp["ModelArtifacts"]["S3ModelArtifacts"].startswith(
            params["OutputDataConfig"]["S3OutputPath"]
        )
    )
    assert_true(training_job_name in (resp["ModelArtifacts"]["S3ModelArtifacts"]))
    assert_true(
        resp["ModelArtifacts"]["S3ModelArtifacts"].endswith("output/model.tar.gz")
    )
    assert_equal(resp["TrainingJobStatus"], "Completed")
    assert_equal(resp["SecondaryStatus"], "Completed")
    assert_equal(resp["HyperParameters"], params["HyperParameters"])
    assert_equal(
        resp["AlgorithmSpecification"]["TrainingImage"],
        params["AlgorithmSpecification"]["TrainingImage"],
    )
    assert_equal(
        resp["AlgorithmSpecification"]["TrainingInputMode"],
        params["AlgorithmSpecification"]["TrainingInputMode"],
    )
    assert_true("MetricDefinitions" in resp["AlgorithmSpecification"])
    assert_true("Name" in resp["AlgorithmSpecification"]["MetricDefinitions"][0])
    assert_true("Regex" in resp["AlgorithmSpecification"]["MetricDefinitions"][0])
    assert_equal(resp["RoleArn"], FAKE_ROLE_ARN)
    assert_equal(resp["InputDataConfig"], params["InputDataConfig"])
    assert_equal(resp["OutputDataConfig"], params["OutputDataConfig"])
    assert_equal(resp["ResourceConfig"], params["ResourceConfig"])
    assert_equal(resp["StoppingCondition"], params["StoppingCondition"])
    assert_true(isinstance(resp["CreationTime"], datetime.datetime))
    assert_true(isinstance(resp["TrainingStartTime"], datetime.datetime))
    assert_true(isinstance(resp["TrainingEndTime"], datetime.datetime))
    assert_true(isinstance(resp["LastModifiedTime"], datetime.datetime))
    assert_true("SecondaryStatusTransitions" in resp)
    assert_true("Status" in resp["SecondaryStatusTransitions"][0])
    assert_true("StartTime" in resp["SecondaryStatusTransitions"][0])
    assert_true("EndTime" in resp["SecondaryStatusTransitions"][0])
    assert_true("StatusMessage" in resp["SecondaryStatusTransitions"][0])
    assert_true("FinalMetricDataList" in resp)
    assert_true("MetricName" in resp["FinalMetricDataList"][0])
    assert_true("Value" in resp["FinalMetricDataList"][0])
    assert_true("Timestamp" in resp["FinalMetricDataList"][0])


@mock_sagemaker
def test_list_training_jobs():
    sagemaker = boto3.client("sagemaker", region_name=TEST_REGION_NAME)

    container = "382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:1"
    bucket = "my-bucket"
    prefix = "sagemaker/DEMO-breast-cancer-prediction/"

    params = {
        "AlgorithmSpecification": {
            "TrainingImage": container,
            "TrainingInputMode": "File",
        },
        "RoleArn": FAKE_ROLE_ARN,
        "OutputDataConfig": {"S3OutputPath": "s3://{}/{}/".format(bucket, prefix)},
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.c4.2xlarge",
            "VolumeSizeInGB": 10,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 60 * 60},
    }

    params["TrainingJobName"] = "Training2ndJob"
    resp = sagemaker.create_training_job(**params)
    assert_equal(resp["ResponseMetadata"]["HTTPStatusCode"], 200)

    time.sleep(1)
    time_after_first_job = datetime.datetime.now(tz=pytz.utc)
    time.sleep(1)

    params["TrainingJobName"] = "TrainingFirstJob"
    resp = sagemaker.create_training_job(**params)
    assert_equal(resp["ResponseMetadata"]["HTTPStatusCode"], 200)

    time.sleep(2)

    params["TrainingJobName"] = "ThirdJob"
    resp = sagemaker.create_training_job(**params)
    assert_equal(resp["ResponseMetadata"]["HTTPStatusCode"], 200)

    resp = sagemaker.list_training_jobs()
    assert_equal(len(resp["TrainingJobSummaries"]), 3)

    resp = sagemaker.list_training_jobs(NameContains="Job")
    assert_equal(len(resp["TrainingJobSummaries"]), 3)

    resp = sagemaker.list_training_jobs(NameContains="Third")
    assert_equal(len(resp["TrainingJobSummaries"]), 1)

    resp = sagemaker.list_training_jobs(NameContains="Training")
    assert_equal(len(resp["TrainingJobSummaries"]), 2)

    resp = sagemaker.list_training_jobs(NameContains="Fourth")
    assert_equal(len(resp["TrainingJobSummaries"]), 0)

    resp = sagemaker.list_training_jobs(StatusEquals="Completed")
    assert_equal(len(resp["TrainingJobSummaries"]), 3)

    resp = sagemaker.list_training_jobs(StatusEquals="Failed")
    assert_equal(len(resp["TrainingJobSummaries"]), 0)

    resp = sagemaker.list_training_jobs(SortBy="Name")
    assert_equal(resp["TrainingJobSummaries"][0]["TrainingJobName"], "ThirdJob")

    resp = sagemaker.list_training_jobs(SortBy="Name", SortOrder="Descending")
    assert_equal(resp["TrainingJobSummaries"][0]["TrainingJobName"], "TrainingFirstJob")

    resp = sagemaker.list_training_jobs(SortBy="CreationTime")
    assert_equal(resp["TrainingJobSummaries"][0]["TrainingJobName"], "Training2ndJob")

    resp = sagemaker.list_training_jobs(SortBy="CreationTime", SortOrder="Descending")
    assert_equal(resp["TrainingJobSummaries"][0]["TrainingJobName"], "ThirdJob")

    resp = sagemaker.list_training_jobs(CreationTimeBefore=time_after_first_job)
    assert_equal(len(resp["TrainingJobSummaries"]), 1)
    assert_equal(resp["TrainingJobSummaries"][0]["TrainingJobName"], "Training2ndJob")

    resp = sagemaker.list_training_jobs(CreationTimeAfter=time_after_first_job)
    assert_equal(len(resp["TrainingJobSummaries"]), 2)
    job_names = [summary["TrainingJobName"] for summary in resp["TrainingJobSummaries"]]
    assert_true(
        all(job_name in ["TrainingFirstJob", "ThirdJob"] for job_name in job_names)
    )

    resp = sagemaker.list_training_jobs(LastModifiedTimeBefore=time_after_first_job)
    assert_equal(len(resp["TrainingJobSummaries"]), 1)
    assert_equal(resp["TrainingJobSummaries"][0]["TrainingJobName"], "Training2ndJob")

    resp = sagemaker.list_training_jobs(LastModifiedTimeAfter=time_after_first_job)
    assert_equal(len(resp["TrainingJobSummaries"]), 2)
    job_names = [summary["TrainingJobName"] for summary in resp["TrainingJobSummaries"]]
    assert_true(
        all(job_name in ["TrainingFirstJob", "ThirdJob"] for job_name in job_names)
    )
