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

![LoRA Architecture](/static/images/lab-1-sft/lora_animated.gif)

### Key Hyperparameters

| Parameter      | Description              | Typical Values |
| -------------- | ------------------------ | -------------- |
| `lora_r`       | Rank of adapter matrices | 4-64           |
| `lora_alpha`   | Scaling factor           | 2-4x the rank  |
| `lora_dropout` | Regularization           | 0.05-0.1       |

---

## Option 1: Customize with Code

::alert[📒 Open the notebook **`lab-1-supervised-fine-tuning/2-fine-tune-llm.ipynb`**]

### Prerequisites

Retrieve the datasets created in the previous step. The base model is defined once in `config.py` and imported here, so every notebook in the lab stays in sync:

```python
from sagemaker.ai_registry.dataset import DataSet
from config import BASE_MODEL_ID

base_model_id = BASE_MODEL_ID  # "huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16"

training_dataset = DataSet.get(name="contractnli-nda-review-train")
val_dataset = DataSet.get(name="contractnli-nda-review-val")
```

:::alert{header="Note" type="info"}
Want to try a different model? The notebook includes a cell that lists all SageMaker JumpStart models supporting customization. To switch, just update `BASE_MODEL_ID` in `config.py` — every notebook in this lab picks up the change automatically. Not every model supports every technique, so a "No recipes found" error means the model doesn't support SFT.
:::

### Create Model Package Group

Model Package Groups organize your fine-tuned model versions. The name is derived from the base model ID with a `-contractnli-sft-mpg` suffix, hash-truncated to SageMaker's 63-character limit when needed. Notebooks 2, 3, 4 and 4a all derive it the same way:

```python
import hashlib
from botocore.exceptions import ClientError
from sagemaker.core.resources import ModelPackageGroup

MAX_MPG_NAME_LENGTH = 63
suffix = "-contractnli-sft-mpg"

candidate = f"{base_model_id}{suffix}"
if len(candidate) > MAX_MPG_NAME_LENGTH:
    digest = hashlib.sha1(base_model_id.encode()).hexdigest()[:6]
    # reserve room for the suffix, a hyphen separator, and the 6-char hash
    keep = MAX_MPG_NAME_LENGTH - len(suffix) - len(digest) - 1
    truncated = base_model_id[:keep].rstrip("-")
    model_package_group_name = f"{truncated}-{digest}{suffix}"
else:
    model_package_group_name = candidate

try:
    model_package_group = ModelPackageGroup.get(
        model_package_group_name=model_package_group_name
    )
    print(f"Model Package Group already exists: {model_package_group_name}")
except ClientError:
    model_package_group = ModelPackageGroup.create(
        model_package_group_name=model_package_group_name,
        model_package_group_description="Store models from SageMaker serverless SFT customization",
    )
    print(f"Created Model Package Group: {model_package_group_name}")
```

### Run Serverless Fine-Tuning Job

Use the `SFTTrainer` class to launch a serverless fine-tuning job:

```python
from sagemaker.train.common import TrainingType
from sagemaker.train.sft_trainer import SFTTrainer

trainer = SFTTrainer(
    model=base_model_id,
    training_type=TrainingType.LORA,
    model_package_group=model_package_group_name,
    training_dataset=training_dataset,
    validation_dataset=val_dataset,
    s3_output_path=output_path,
    sagemaker_session=sess,
    role=role,
    accept_eula=True,
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
trainer.hyperparameters.learning_rate = 0.0001
trainer.hyperparameters.global_batch_size = 64
trainer.hyperparameters.max_epochs = 10
trainer.hyperparameters.lr_warmup_steps_ratio = 0.1
trainer.hyperparameters.lora_rank = 32
trainer.hyperparameters.lora_alpha = 64
```

:::alert{header="Note" type="warning"}
**Count optimizer steps, not epochs.** With `global_batch_size = 64`, this dataset gives only five gradient steps per epoch, so the default `max_epochs = 1` would learn almost nothing. Ten epochs gets to 50 steps. With a larger dataset the same 50 steps arrive in one or two epochs, and you would set this number back down.

Ten is not arbitrary. The output format and the label distribution are learned early, but `evidence_f1` — citing the clause a reviewer would actually check — keeps improving well after accuracy flattens, and it is the metric this task is judged on. Shorter runs leave it short of the frontier baseline measured in the next section.
:::

### Submit the Training Job

```python
training_job = trainer.train(wait=False)

TRAINING_JOB_NAME = training_job.training_job_name
pprint(training_job)
```

:::alert{header="Important" type="warning"}
The serverless fine-tuning job takes **20-25 minutes** for the 10 epochs configured above. SageMaker AI provisions the compute for you, and that share is fixed, so the wall-clock does not scale with epochs the way you might expect — halving the epochs does not halve the time. With a larger dataset the training share grows while the start-up cost stays roughly where it is.
:::

### Monitor Training Progress

Navigate to **Jobs** → **Training** in SageMaker Studio to view your training jobs:

![Studio Training Jobs](/static/images/lab-1-sft/studio-jobs-training.png)

Click on a job to view details, metrics, and logs.

### Training Metrics with MLflow

With serverless customization, training metrics are automatically logged to MLflow. You can view training and validation loss directly in SageMaker Studio:

![MLflow Training Metrics](/static/images/lab-1-sft/studio-mlflow-metrics.png)

The Performance tab shows:

- **Training Loss** - Decreasing loss indicates the model is learning
- **Validation Loss** - Monitors overfitting on held-out data

Click **View full metrics in MLflow** for detailed experiment tracking and visualization.

## Option 2: Customize with UI

You can also run serverless customization directly from the SageMaker Studio UI without writing code.

In the **Models** section, find your model and click **Customize** → **Customize with UI**:

![Studio Models Customize](/static/images/lab-1-sft/studio-models-customize.png)

This opens the customization wizard where you can:

- Select the customization technique (Supervised Fine-Tuning, DPO, RLVR, RLAIF)
- Upload or select existing datasets
- Configure hyperparameters (batch size, learning rate, epochs)
- Set the output S3 location

![Studio Customize UI](/static/images/lab-1-sft/studio-customize-ui.png)

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach for more flexibility and reproducibility. The UI is great for quick experiments and prototyping.
:::

:::alert{header="Short on time?" type="warning"}
If you don't want to wait for training to complete, you can use notebook **`2b-import-model-from-s3.ipynb`** to download a pre-trained checkpoint, upload it to your S3 bucket, and register it as a Model Package directly in the SageMaker Model Registry. This lets you skip ahead to evaluation and deployment immediately.
:::

---
