---
title: "Fine-Tuning"
weight: 2
---

## Understanding LoRA

Low-Rank Adaptation (LoRA) is an efficient fine-tuning technique that:

- Keeps the original model weights **frozen**
- Learns small adapter matrices instead of updating all parameters
- Reduces trainable parameters by **10-100x**
- Enables fine-tuning with significantly less memory

![LoRA Architecture](/static/images/lab-2-dpo/lora_animated.gif)

### Key Hyperparameters

| Parameter      | Description              | Typical Values |
| -------------- | ------------------------ | -------------- |
| `lora_r`       | Rank of adapter matrices | 4-64           |
| `lora_alpha`   | Scaling factor           | 2-4x the rank  |
| `lora_dropout` | Regularization           | 0.05-0.1       |

---

## Alternative: Customize with UI

You can also run serverless customization directly from the SageMaker Studio UI without writing code.

In the **Models** section, find your model and click **Customize** → **Customize with UI**:

![Studio Models Customize](/static/images/lab-2-dpo/studio-models-customize.png)

This opens the customization wizard where you can:

- Select the customization technique (Supervised Fine-Tuning, DPO, RLVR, RLAIF)
- Upload or select existing datasets
- Configure hyperparameters (batch size, learning rate, epochs)
- Set the output S3 location

![Studio Customize UI](/static/images/lab-2-dpo/studio-customize-ui.png)

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach for more flexibility and reproducibility. The UI is great for quick experiments and prototyping.
:::

---

## Customize with Code

::alert[📒 Open the notebook **`lab-2-direct-preference-optimization-DPO/2-dpo-trainer.ipynb`**]

## Prerequisites

Retrieve the datasets created in the previous step:

```python
from sagemaker.ai_registry.dataset import DataSet

base_model_id = "meta-textgeneration-llama-3-2-1b-instruct"

training_dataset = DataSet.get(name="humanlike-dpo-train")
validation_dataset = DataSet.get(name="humanlike-dpo-val")
```

## Create Model Package Group

Model Package Groups organize your fine-tuned model versions:

```python
from sagemaker.core.resources import ModelPackageGroup

model_package_group_name = f"{base_model_id}-dpo"

model_package_group = ModelPackageGroup.create(
    model_package_group_name=model_package_group_name,
    model_package_group_description="Store models from SageMaker serverless DPO customization",
)
```

## Run Serverless Fine-Tuning Job

Use the `DPOTrainer` class to launch a serverless fine-tuning job:

#### Create DPO Trainer (Direct Preference Optimization)

Direct Preference Optimization (DPO) is a method for training language models to follow human preferences. Unlike traditional RLHF (Reinforcement Learning from Human Feedback), DPO directly optimizes the model using preference pairs without needing a reward model.

**Key Benefits:**

- Simpler than RLHF - no reward model required
- More stable training process
- Direct optimization on preference data
- Works with LoRA for efficient fine-tuning

##### Key Parameters:

- `model` Base model to fine-tune (from SageMaker Hub)
- `training_type` Fine-tuning method (LoRA recommended for efficiency)
- `training_dataset` ARN of the registered preference dataset. Training Dataset - either Dataset ARN or S3 Path of the dataset (Please note these are required for a training job to run, can be either provided via Trainer or `.train()`)
- `model_package_group` Where to store the fine-tuned model
- `mlflow_experiment_name`: MLFlow app experiment name (str) (optional)
- `mlflow_run_name`: MLFlow app run name (str) (optional)
- `validation_dataset`: Validation Dataset - either Dataset ARN or S3 Path of the dataset (optional)
- `s3_output_path`: S3 path for the trained model artifacts (optional)

```python
from sagemaker.train.common import TrainingType
from sagemaker.train.dpo_trainer import DPOTrainer

trainer = DPOTrainer(
    model=base_model_id,
    training_type=TrainingType.LORA,
    model_package_group=model_package_group,
    training_dataset=training_dataset,
    validation_dataset=validation_dataset,
    s3_output_path=output_path,
    mlflow_experiment_name=mlflow_experiment_name,
    base_job_name=job_name,
    sagemaker_session=sess,
    accept_eula=True,
    role=role
)

```

### View Default Hyperparameters

```python
from rich.pretty import pprint

print("Default Finetuning options:")
pprint(trainer.hyperparameters.to_dict())
```

### Override Hyperparameters

Customize the training configuration:

```python
trainer.hyperparameters.global_batch_size = 64
trainer.hyperparameters.max_epochs = 1
```

### Submit the Training Job

```python
training_job = trainer.train(wait=False)

TRAINING_JOB_NAME = training_job.training_job_name
pprint(training_job)
```

:::alert{header="Important" type="warning"}
The serverless fine-tuning job can take **30-45 minutes** to complete. SageMaker AI automatically provisions the appropriate compute resources based on the model and data size.
:::

## Monitor Training Progress

Navigate to **Jobs** → **Training** in SageMaker Studio to view your training jobs:

![Studio Training Jobs](/static/images/lab-2-dpo/studio-jobs-training.png)

Click on a job to view details, metrics, and logs.

### Training Metrics with MLflow

With serverless customization, training metrics are automatically logged to MLflow. You can view training and validation loss directly in SageMaker Studio:

![MLflow Training Metrics](/static/images/lab-2-dpo/studio-mlflow-metrics.png)

The Performance tab shows:

- **Training Loss** - Decreasing loss indicates the model is learning
- **Reward Chosen** - Stabilizing reward for preferred responses shows the model is consistently scoring good outputs.
- **Reward Rejected** - Sharply declining reward for rejected responses shows the model is increasingly penalizing bad outputs.
- **Reward Margin** - Growing gap between chosen and rejected rewards shows the model is better distinguishing good from bad responses.
- **Reward Accuracy** - Reaching ~1.0 shows the model almost always prefers the chosen response over the rejected one.

Click **View full metrics in MLflow** for detailed experiment tracking and visualization.

:::alert{header="Short on time?" type="warning"}
If you don't want to wait for training to complete, you can use notebook **`2b-dpo-import-model-from-s3.ipynb`** to download a pre-trained checkpoint, upload it to your S3 bucket, and register it as a Model Package directly in the SageMaker Model Registry. This lets you skip ahead to evaluation and deployment immediately.
:::
