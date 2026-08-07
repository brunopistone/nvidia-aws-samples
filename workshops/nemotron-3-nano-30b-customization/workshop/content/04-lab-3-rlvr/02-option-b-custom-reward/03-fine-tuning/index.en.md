---
title: "Fine-Tuning"
weight: 3
---

This step launches a serverless RLVR customization job that uses the GSM8K datasets from notebook 1 and the custom Reward Function evaluator from notebook 2. The only structural difference from Option A is that we pass our own evaluator to the trainer via the `custom_reward_function` parameter.

---

## Customize with Code

::alert[📒 Open the notebook **`lab-3a-custom-reward-function-rlvr/3-train-with-custom-reward-function.ipynb`**]

### Choosing a base model

The base model for this lab is **Qwen 3 0.6B** (`huggingface-reasoning-qwen3-06b`) — a small reasoning model chosen for fast, low-cost workshop runs. You can browse other customization-capable JumpStart models from the notebook and switch by updating the `base_model_id`.

:::alert{header="Note" type="info"}
Not every JumpStart model supports every customization technique. If you hit a "No recipes found" error, that model may not support RLVR.
:::

### Retrieve datasets and the reward function evaluator

```python
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.evaluator import Evaluator

base_model_id = "huggingface-reasoning-qwen3-06b"
training_dataset = DataSet.get(name=f"{project_prefix}-train")
validation_dataset = DataSet.get(name=f"{project_prefix}-val")
custom_reward_function = Evaluator.get(name=reward_function_name)
```

### Create Model Package Group

```python
from botocore.exceptions import ClientError
from sagemaker.core.resources import ModelPackageGroup

model_package_group_name = f"{base_model_id}-custom-rw-rlvr"
try:
    model_package_group = ModelPackageGroup.get(model_package_group_name)
except ClientError:
    model_package_group = ModelPackageGroup.create(
        model_package_group_name=model_package_group_name,
        model_package_group_description="Store models from SageMaker RLVR customization with a custom reward function",
    )
```

### Configure the RLVR Trainer

The `RLVRTrainer` accepts your registered Reward Function evaluator through the `custom_reward_function` parameter:

```python
from sagemaker.train.rlvr_trainer import RLVRTrainer

trainer = RLVRTrainer(
    model=base_model_id,
    model_package_group=model_package_group,
    custom_reward_function=custom_reward_function,
    training_dataset=training_dataset,
    validation_dataset=validation_dataset,
    s3_output_path=output_path,
    mlflow_experiment_name=mlflow_experiment_name,
    base_job_name=job_name,
    sagemaker_session=sess,
    accept_eula=True,
    role=role,
)
```

| Parameter                | Description                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `model`                  | Base model ID from the SageMaker Hub                                                                 |
| `custom_reward_function` | **The key difference vs Option A** — your registered Reward Function `Evaluator` (or its ARN string) |
| `training_dataset`       | Registered training dataset from the AI Registry                                                     |
| `validation_dataset`     | Registered validation dataset from the AI Registry                                                   |
| `model_package_group`    | Where to store the fine-tuned model versions                                                         |
| `s3_output_path`         | S3 location for model artifacts                                                                      |
| `mlflow_experiment_name` | MLflow experiment name for metric tracking                                                           |

:::alert{header="Custom vs built-in reward" type="info"}
In Option A, the trainer fell back to the built-in `default_compute_score`. Here, passing `custom_reward_function` tells SageMaker to invoke your registered evaluator to score each rollout during training.
:::

### Inspect and adjust hyperparameters

```python
from rich.pretty import pprint

print("Default finetuning options:")
pprint(trainer.hyperparameters.to_dict())
```

```python
trainer.hyperparameters.max_epochs = 1
trainer.hyperparameters.learning_rate = 5e-06
trainer.hyperparameters.global_batch_size = 128
trainer.hyperparameters.max_prompt_length = 1024
```

### Submit the Training Job

```python
training_job = trainer.train(wait=False)
print(f"Training job: {training_job.training_job_name}")
```

:::alert{header="Important" type="warning"}
The serverless RLVR training job can take **15-20 minutes** to complete. SageMaker AI automatically provisions the appropriate compute resources based on the model and data size.
:::

### Monitor Training Progress

Navigate to **Jobs** → **Training** in SageMaker Studio to view your training jobs:

![Studio Training Jobs](/static/images/lab-3-rlvr/studio-jobs-training.png)

Click on a job to view details, metrics, and logs.

### Training Metrics with MLflow

Training metrics are automatically logged to MLflow. In addition to the standard RLVR metrics, you'll see the custom metrics your reward function emits (`numeric_correctness`, `answer_parseable`, `final_answer_format`, `reasoning_present`), giving you insight into _why_ the reward changes over time:

![MLflow Training Metrics](/static/images/lab-3-rlvr/studio-mlflow-metrics.png)

Click **View full metrics in MLflow** for detailed experiment tracking and visualization.

:::alert{header="Short on time?" type="warning"}
If you don't want to wait for training to complete, you can use notebook **`3b-import-model-from-s3.ipynb`** to download a pre-trained checkpoint, upload it to your S3 bucket, and register it as a Model Package directly in the SageMaker Model Registry. This lets you skip ahead to evaluation and deployment immediately.
:::

---

Once training completes, you're ready to evaluate the fine-tuned model.
