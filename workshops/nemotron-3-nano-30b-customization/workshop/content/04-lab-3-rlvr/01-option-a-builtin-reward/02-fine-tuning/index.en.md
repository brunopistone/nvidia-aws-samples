---
title: "Fine-Tuning"
weight: 2
---

## What is RLVR?

Reinforcement Learning from Verifiable Rewards (RLVR) is a method for training language models on tasks with clear, verifiable answers. Unlike RLHF which requires human feedback, RLVR uses automatic verification of correctness.

**Key Benefits:**

- No human feedback required — uses automatic verification
- Works well for math, coding, and other verifiable tasks
- Scalable training without a separate reward model
- Direct optimization on correctness

---

## Option 1: Customize with Code

::alert[📒 Open the notebook **`lab-3-reinforcement-learning-from-verifiable-rewards/2-trainer.ipynb`**]

### Prerequisites

Retrieve the datasets created in the previous step:

```python
from sagemaker.ai_registry.dataset import DataSet

base_model_id = "huggingface-reasoning-qwen3-06b"
training_dataset = DataSet.get(name="rlvr-train")
validation_dataset = DataSet.get(name="rlvr-val")
```

### Create Model Package Group

Model Package Groups organize your fine-tuned model versions:

```python
from sagemaker.core.resources import ModelPackageGroup

model_package_group_name = f"{base_model_id}-rlvr"

model_package_group = ModelPackageGroup.create(
    model_package_group_name=model_package_group_name,
    model_package_group_description="Store models from SageMaker serverless RLVR customization",
)
```

### Configure the RLVR Trainer

The `RLVRTrainer` is the high-level API for serverless RLVR training:

```python
from sagemaker.train.rlvr_trainer import RLVRTrainer

trainer = RLVRTrainer(
    model=base_model_id,
    model_package_group=model_package_group,
    training_dataset=training_dataset,
    validation_dataset=validation_dataset,
    s3_output_path=output_path,
    mlflow_experiment_name="qwen3-06b-rlvr",
    base_job_name=job_name,
    sagemaker_session=sess,
    accept_eula=True,
    role=role,
)
```

| Parameter                | Description                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `model`                  | Base model ID from the SageMaker Hub                                                                              |
| `training_dataset`       | Registered training dataset from the AI Registry                                                                  |
| `validation_dataset`     | Registered validation dataset from the AI Registry                                                                |
| `model_package_group`    | Where to store the fine-tuned model versions                                                                      |
| `s3_output_path`         | S3 location for model artifacts                                                                                   |
| `mlflow_experiment_name` | MLflow experiment name for metric tracking                                                                        |
| `custom_reward_function` | _(Optional)_ Custom reward function — an Evaluator ARN string or `Evaluator` object (e.g., a Lambda-based reward) |

> **Note:** Setting `accept_eula=True` is required for gated models.

:::alert{header="Reward Function" type="info"}
The `RLVRTrainer` uses `default_compute_score` (exact match) as its reward function, which is sufficient for tasks like GSM8K where answers are verifiable. If you need custom reward logic (e.g., a Lambda function), you can pass a `custom_reward_function` parameter — see the table above.
:::

### View Default Hyperparameters

```python
from rich.pretty import pprint

print("Default Finetuning options:")
pprint(trainer.hyperparameters.to_dict())
```

### Override Hyperparameters

Customize the training configuration:

```python
trainer.hyperparameters.max_epochs = 1
trainer.hyperparameters.learning_rate = 5e-06
trainer.hyperparameters.global_batch_size = 128
trainer.hyperparameters.max_prompt_length = 1024
```

### Submit the Training Job

```python
training_job = trainer.train(wait=False)

TRAINING_JOB_NAME = training_job.training_job_name
pprint(training_job)
```

:::alert{header="Important" type="warning"}
The serverless RLVR training job can take **15-20 minutes** to complete. SageMaker AI automatically provisions the appropriate compute resources based on the model and data size.
:::

### Monitor Training Progress

Navigate to **Jobs** → **Training** in SageMaker Studio to view your training jobs:

![Studio Training Jobs](/static/images/lab-3-rlvr/studio-jobs-training.png)

Click on a job to view details, metrics, and logs.

### Training Metrics with MLflow

With serverless customization, training metrics are automatically logged to MLflow. You can view training metrics directly in SageMaker Studio:

![MLflow Training Metrics](/static/images/lab-3-rlvr/studio-mlflow-metrics.png)

Click **View full metrics in MLflow** for detailed experiment tracking and visualization.

## Option 2: Customize with UI

You can also run serverless customization directly from the SageMaker Studio UI without writing code.

In the **Models** section, find your model and click **Customize** → **Customize with UI**:

![Studio Models Customize](/static/images/lab-3-rlvr/studio-models-customize.png)

This opens the customization wizard where you can:

- Select **RLVR** as the customization technique
- Upload or select existing datasets
- Configure hyperparameters (batch size, epochs)
- Set the output S3 location

![Studio Customize UI](/static/images/lab-3-rlvr/studio-customize-ui.png)

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach for more flexibility and reproducibility. The UI is great for quick experiments and prototyping.
:::

:::alert{header="Short on time?" type="warning"}
If you don't want to wait for training to complete, you can use notebook **`2b-import-model-from-s3.ipynb`** to download a pre-trained checkpoint, upload it to your S3 bucket, and register it as a Model Package directly in the SageMaker Model Registry. This lets you skip ahead to evaluation and deployment immediately.
:::

---
