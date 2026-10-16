---
title: "Fine-tuning"
weight: 2
---

## What you will learn

You will configure the SageMaker serverless `SFTTrainer`, connect registered datasets to a Model Package Group, interpret the LoRA and optimization settings, submit an asynchronous customization job, and verify the resulting model lineage. The objective is to adapt contract decisions and citations, not merely to make training loss decrease.

::alert[Open [01-serverless-workshop/2-fine-tune-llm.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/2-fine-tune-llm.ipynb) after [data preparation](/02-serverless/01-data-preparation/). This notebook's SageMaker `SFTTrainer` is different from the TRL class with the same name used inside the 4B Training jobs journey.]{type="info"}

## LoRA: what changes and what stays frozen

For a weight matrix `W` with shape `output_dim * input_dim`, ordinary LoRA represents an update using two smaller matrices: `W_updated = W + (alpha / rank) * B * A`, with `A` shaped `rank * input_dim` and `B` shaped `output_dim * rank`. The base matrix remains frozen during adapter training. The trainable update has `rank * (input_dim + output_dim)` parameters instead of `input_dim * output_dim` for a full matrix update.

![LoRA architecture](/static/images/lab-1-sft/lora_animated.gif)

Rank controls the size of the low-rank update. Alpha controls its scaling in the ordinary LoRA formulation; neither parameter is a direct measure of task quality. Fewer trainable weights reduce adapter and optimizer-state requirements, but base weights, activations, and temporary tensors still consume memory. Likewise, the 30B-A3B model's active-parameter designation does not mean that only those active parameters need to be stored for serving.

The serverless trainer resolves a managed recipe for the selected model. You do not supply a training script, a `Torchrun` launcher, or an instance type in this track. You still own input quality, supported configuration choices, access permissions, evaluation, and cost control.

## 1. Confirm the model and available recipe

The executable [config.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/config.py) sets the following JumpStart identifier:

```python
from config import BASE_MODEL_ID, DATASET_PREFIX

base_model_id = BASE_MODEL_ID
assert base_model_id == "huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16"
```

The assertion is an optional workshop checkpoint, not a request to override your configuration silently. The notebook's Hub listing paginates through `SageMakerPublicHub` and prints entries marked with `@capability:customization`. That is discovery, not proof that every listed model supports this SFT technique or the later serving image. If recipe resolution fails, check the actual model identifier, Region, SDK, and supported technique before changing data or hyperparameters.

Before constructing the trainer, verify the [account prerequisites](/01-prerequisites/2-account/): customization and registry access, permission to pass the execution role, S3 access to inputs and outputs, and acceptance of the model's license terms. The notebook passes `accept_eula=True`; run that cell only after reviewing the terms.

## 2. Resolve data and the output destination

Run the session setup first. It initializes `sess`, `role`, `bucket_name`, `default_prefix`, and the AWS clients, then resolves and prints a ready `mlflow_app_arn`. A configured `MLFLOW_APP_ARN` in `config.py` is reused and waited for; invalid or inaccessible explicit ARNs fail without replacement. An empty ARN first reuses the uniquely matching tagged `workshop-mlflow-<stack UUID>` app, ahead of any named workshop duplicate. Ambiguous matches stop setup and request an explicit ARN; pending apps are waited for, and failed apps are not replaced. Only if no workshop app matches does setup use the pinned SDK's native discovery/create path, with `AccountDefaultStatus="DISABLED"` on creation and no domain requirement. Setting both creation overrides selects named creation instead. No manual ARN or stack update is required for discovery. Save the resolved ARN for evaluation; see [MLflow setup behavior](/01-prerequisites/3-sagemaker/#7-reuse-or-create-the-workshop-mlflow-app). The trainer receives this ARN explicitly and records runs under `contractnli-serverless-sft`. Then the notebook retrieves the named training and validation datasets:

```python
from sagemaker.ai_registry.dataset import DataSet

training_dataset = DataSet.get(name=f"{DATASET_PREFIX}-train")
val_dataset = DataSet.get(name=f"{DATASET_PREFIX}-val")
output_path = (f"s3://{bucket_name}/{default_prefix}/{base_model_id}-contractnli"
               if default_prefix else f"s3://{bucket_name}/{base_model_id}-contractnli")
print(output_path)
```

These are registered dataset objects, not the lists of records from the previous kernel. Confirm the resolved versions and S3 locations belong to this preparation run. A bare-name lookup and a fixed S3 key can both select something newer than the data you intended to use. The test dataset must not be substituted for validation: it is reserved for the held-out comparison.

## 3. Create the package group used for handoff

A Model Package Group organizes versions of the customized model. Evaluation and deployment derive the same name; do not independently shorten it in one notebook. The notebook truncates long names and includes a short hash to keep the name within its 63-character bound.

```python
import hashlib

MAX_MPG_NAME_LENGTH = 63
suffix = "-contractnli-sft-mpg"
candidate = f"{base_model_id}{suffix}"
if len(candidate) > MAX_MPG_NAME_LENGTH:
    digest = hashlib.sha1(base_model_id.encode()).hexdigest()[:6]
    keep = MAX_MPG_NAME_LENGTH - len(suffix) - len(digest) - 1
    truncated = base_model_id[:keep].rstrip("-")
    model_package_group_name = f"{truncated}-{digest}{suffix}"
else:
    model_package_group_name = candidate
print(model_package_group_name)
```

Run the subsequent `ModelPackageGroup.get`/`create` cell. An existing group is normal on a repeat experiment. However, the notebook catches a general `ClientError` before attempting creation; access denied and not found are not the same condition. Inspect the original error if creation fails instead of assuming the group was absent.

## 4. Construct and inspect the trainer

```python
from sagemaker.train.common import TrainingType
from sagemaker.train.sft_trainer import SFTTrainer

MAX_JOB_NAME_LENGTH, TIMESTAMP_LENGTH = 63, 15
base_job_name = "contractnli-sft"[:MAX_JOB_NAME_LENGTH - TIMESTAMP_LENGTH].rstrip("-")

trainer = SFTTrainer(
    model=base_model_id,
    training_type=TrainingType.LORA,
    model_package_group=model_package_group_name,
    mlflow_resource_arn=mlflow_app_arn,
    mlflow_experiment_name="contractnli-serverless-sft",
    training_dataset=training_dataset,
    validation_dataset=val_dataset,
    s3_output_path=output_path,
    sagemaker_session=sess,
    role=role,
    accept_eula=True,
    base_job_name=base_job_name,
)

for key, value in trainer.hyperparameters.to_dict().items():
    print(key, value)
```

Read the resolved defaults before editing them. They are the supported configuration surface of the selected recipe, not a generic TRL configuration. Preserve the printed configuration with your run notes so that an unexplained result can be traced back to the actual model, data versions, and settings.

```python
trainer.hyperparameters.global_batch_size = 64
trainer.hyperparameters.max_epochs = 10
trainer.hyperparameters.learning_rate = 0.0001
trainer.hyperparameters.lr_warmup_steps_ratio = 0.1
trainer.hyperparameters.lora_rank = 32
trainer.hyperparameters.lora_alpha = 64
```

| Override                      | What it controls                                 | What to watch                                         |
| ----------------------------- | ------------------------------------------------ | ----------------------------------------------------- |
| `global_batch_size = 64`      | Effective batch targeted by the managed recipe   | Actual retained data and batching in logs             |
| `max_epochs = 10`             | Maximum passes over the effective training data  | Overfitting and validation behavior on a small corpus |
| `learning_rate = 0.0001`      | Update magnitude under the scheduler             | Unstable loss or little adaptation                    |
| `lr_warmup_steps_ratio = 0.1` | Warmup fraction in the recipe's scheduling logic | Resolved scheduler and update count                   |
| `lora_rank = 32`              | Adapter rank                                     | Capacity and trainable-parameter footprint            |
| `lora_alpha = 64`             | Adapter scaling                                  | Keep rank and scaling recorded together               |

An epoch is a pass through the data the trainer actually uses. An optimizer step is an update after batching and any accumulation. With a dataset smaller than a large effective batch schedule might suggest, ten epochs can still represent relatively few updates. Do not equate ten epochs here with ten epochs under the 4B Training jobs journey's different batching and loss pipeline.

### Do not treat the step-count demonstration as telemetry

The notebook's arithmetic cell fixes `records = 423` and `over_limit = 108`; it does not tokenize this run's data or read its logs. Other notebook prose describes different sequence caps. Consequently, its printed retained count, step count, and schedule calculation are illustrations from prior settings, not evidence about this 30B-A3B run. Inspect actual preprocessing, resolved limits, and training logs before reporting coverage. Truncating a record and dropping it entirely have different effects on supervision.

The intended SFT boundary is the completion: the contract and checklist provide context, while the JSON answer provides targets. Because this path delegates training internals to a managed recipe, verify loss-mask and preprocessing behavior through the resolved recipe and available logs rather than copying the 4B Training jobs journey's `completion_only_loss` YAML into this API.

## 5. Submit and monitor remotely

```python
training_job = trainer.train(wait=False)
TRAINING_JOB_NAME = training_job.training_job_name
print(f"launched: {TRAINING_JOB_NAME}")
```

The returned object and printed name mean that submission occurred, not that the weights are ready. Save that exact job name. Follow the job in Studio under **Jobs**, **Training**, or use the notebook's polling cell:

```python
import time

started = time.monotonic()
while True:
    job = sm_client.describe_training_job(TrainingJobName=TRAINING_JOB_NAME)
    status, secondary = job["TrainingJobStatus"], job["SecondaryStatus"]
    print(f"[{(time.monotonic() - started) / 60:5.1f} min] {status} / {secondary}")
    if status in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(60)

if status != "Completed":
    reason = job.get("FailureReason") or "No failure reason reported; inspect CloudWatch logs."
    raise RuntimeError(f"{TRAINING_JOB_NAME}: {status} / {secondary}: {reason}")
print("Completed. Continue to notebook 3 for evaluation.")
```

![Studio training jobs](/static/images/lab-1-sft/studio-jobs-training.png)

Expected output is a sequence of primary and secondary status strings every 60 seconds, ending in a terminal state. The cell raises for `Failed` or `Stopped` and prints completion only for `Completed`. Before polling, the notebook prints direct job-console and CloudWatch links plus a signed MLflow App URL; open the latter within five minutes, or rerun only the link cell for a fresh URL, not training submission. Do not share signed URLs. Inspect logs for preprocessing counts, training/validation loss, saving, and registration. A decreasing training loss does not prove improved contradiction detection or correct citations.

No duration or price is guaranteed. Provisioning, preprocessing, training, saving, and service limits all contribute to elapsed time. Closing the notebook does not stop the job. If you do not intend to finish a submitted run, stop it through the service and verify the terminal state.

### Progress activities: Studio, CloudWatch, and MLflow

While the job runs, open **Studio > Jobs > Training** and select `TRAINING_JOB_NAME`. In the [SageMaker AI console](https://console.aws.amazon.com/sagemaker/), the equivalent navigation is **Training > Training jobs**. Compare the job's input data and resolved configuration with the registry versions you recorded. Follow **View logs**, or open [CloudWatch](https://console.aws.amazon.com/cloudwatch/) under **Logs > Log groups > `/aws/sagemaker/TrainingJobs`**, filtering streams by the exact name and this run's time window. Identify preprocessing, optimizer updates, validation, and export/registration stages.

In Studio, open the **MLflow** application and select the **MLflow App** matching the resolved `mlflow_app_arn` printed during setup, whether reused or newly created. Alternatively use the notebook's temporary app link. Open **Experiments > `contractnli-serverless-sft`** and inspect the current run. Record its ID and compare available training/validation metrics with the logs. Do not create a legacy Tracking Server or another domain. If the app is not visible, verify the resolved ARN, account, Region, and permissions. The app requires no domain association even though Studio provides a convenient UI launch surface.

**Activity:** find one configuration value in the job details and one metric in the run. Explain why a lower loss is not proof of a better evidence citation. If no metric has appeared, use the current CloudWatch stage to distinguish setup from a failure rather than submitting a duplicate job.

## 6. Verify the package, not just the job

After `Completed`, use the package lookup cell in evaluation to identify the resulting version:

```python
response = sm_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)
assert response["ModelPackageSummaryList"], "No model package found"
model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
print(model_package_arn)
```

This returns the newest package in the group, not necessarily a package created by the job you just watched. Verify its lineage against the job name before continuing, especially in shared environments. Record the package ARN alongside input dataset versions, output prefix, and resolved hyperparameters. The deployment lesson later resolves a merged checkpoint under that package's model data source.

| Symptom                                  | Targeted investigation                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------------------------- |
| No SFT recipe found                      | Confirm configured JumpStart ID, technique, Region, and SDK support                     |
| Training cannot read input               | Check dataset import state and the execution role's S3/KMS access                       |
| Unexpectedly small effective dataset     | Read preprocessing limits and logs; do not trust hard-coded notebook counts             |
| `Failed` after training appears finished | Inspect export and registration stages, not just the loss plot                          |
| A package already exists                 | Verify job lineage and version; an older successful run is not evidence for the new one |

Continue to [Evaluation](/02-serverless/03-evaluation/) only with an identified completed run and its intended Model Package. There is no checkpoint-import shortcut notebook in this track's current source directory.
