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

| Parameter    | Description              | Typical Values |
| ------------ | ------------------------ | -------------- |
| `lora_rank`  | Rank of adapter matrices | 4-64           |
| `lora_alpha` | Scaling factor           | 2-4x the rank  |

---

## Option 1: Customize with Code

::alert[📒 Open the notebook **`code/2-fine-tune-llm.ipynb`**]

### Choosing a base model

The base model is configured once in `code/config.py` and picked up by every notebook in the lab:

```python
BASE_MODEL_ID = "huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16"
```

To try a different model, update `BASE_MODEL_ID` — all notebooks pick up the change automatically. The first cell of the notebook lists every JumpStart model that supports customization:

```python
sm = boto3.client("sagemaker")
kwargs = {"HubName": "SageMakerPublicHub", "HubContentType": "Model", "MaxResults": 100}
# ... filters on the "@capability:customization" search keyword
```

:::alert{header="Note" type="info"}
Not every JumpStart model supports every customization technique (SFT, DPO, RLVR, ...). If you hit a "No recipes found" error, that model does not support the technique used in this lab.
:::

### Prerequisites

Retrieve the datasets created in the previous step:

```python
from sagemaker.ai_registry.dataset import DataSet
from config import BASE_MODEL_ID

base_model_id = BASE_MODEL_ID

training_dataset = DataSet.get(name="Multilingual-Thinking-sft-train")
val_dataset = DataSet.get(name="Multilingual-Thinking-sft-val")
```

### Create Model Package Group

Model Package Groups organize your fine-tuned model versions. Because `BASE_MODEL_ID` is long, the notebook hash-truncates the name to stay inside the SageMaker 63-character limit:

```python
import hashlib
from botocore.exceptions import ClientError
from sagemaker.core.resources import ModelPackageGroup

MAX_MPG_NAME_LENGTH = 63
suffix = "-sft-mpg"

candidate = f"{base_model_id}{suffix}"
if len(candidate) > MAX_MPG_NAME_LENGTH:
    digest = hashlib.sha1(base_model_id.encode()).hexdigest()[:6]
    keep = MAX_MPG_NAME_LENGTH - len(suffix) - len(digest) - 1
    model_package_group_name = f"{base_model_id[:keep].rstrip('-')}-{digest}{suffix}"
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

:::alert{header="Keep this name consistent" type="warning"}
The evaluation and deployment notebooks rebuild this exact name to find your fine-tuned model. If you change the suffix or the truncation logic here, change it in `3-evaluation.ipynb`, `4-deployment.ipynb` and `4a-deployment-bedrock.ipynb` too.
:::

Once created, the Model Package Group appears under **Models** in SageMaker Studio. Your fine-tuned model versions are registered under this group after the training job completes.

### Run Serverless Fine-Tuning Job

Use the `SFTTrainer` class to configure a serverless fine-tuning job:

```python
from sagemaker.train.common import TrainingType
from sagemaker.train.sft_trainer import SFTTrainer

# SageMaker appends a "-YYYYMMDDHHMMSS" timestamp and then truncates to 63 chars.
# Pass a short base_job_name so the timestamp survives and every run is unique.
MAX_JOB_NAME_LENGTH = 63
TIMESTAMP_LENGTH = 15
base_job_name = f"{base_model_id}-sft"[: MAX_JOB_NAME_LENGTH - TIMESTAMP_LENGTH].rstrip("-")

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
    base_job_name=base_job_name,
)
```

### View Default Hyperparameters

```python
from rich.pretty import pprint

print("Default Finetuning options:")
pprint(trainer.hyperparameters.to_dict())
```

### Override Hyperparameters

The notebook overrides the defaults with values tuned for this small dataset:

```python
trainer.hyperparameters.learning_rate = 0.0002
trainer.hyperparameters.global_batch_size = 128
trainer.hyperparameters.max_epochs = 10
trainer.hyperparameters.warmup_steps = 6
trainer.hyperparameters.weight_decay = 0.05
trainer.hyperparameters.lora_rank = 32
trainer.hyperparameters.lora_alpha = 64
```

| Hyperparameter      | Value  | Why                                                                  |
| ------------------- | ------ | -------------------------------------------------------------------- |
| `learning_rate`     | `2e-4` | Typical for LoRA, which needs a higher LR than full fine-tuning      |
| `global_batch_size` | `128`  | Serverless model customization enforces a **minimum of 128**         |
| `max_epochs`        | `10`   | The dataset is small (~700 training rows), so more passes are needed |
| `warmup_steps`      | `6`    | Short warmup to stabilise the first updates                          |
| `weight_decay`      | `0.05` | Regularisation to limit overfitting on a small dataset               |
| `lora_rank`         | `32`   | Adapter capacity                                                     |
| `lora_alpha`        | `64`   | 2x the rank, a common default                                        |

:::alert{header="Minimum batch size" type="info"}
Serverless model customization enforces a minimum `global_batch_size` of **128**. Lower values are rejected when the job is submitted.
:::

### Submit the Training Job

```python
training_job = trainer.train(wait=False)

TRAINING_JOB_NAME = training_job.training_job_name
pprint(training_job)
```

`wait=False` submits the job asynchronously and returns immediately.

:::alert{header="Important" type="warning"}
With this dataset and these hyperparameters the job takes roughly **20-30 minutes**. SageMaker AI automatically provisions the appropriate compute based on the model and data size — you don't pick an instance type.
:::

### Monitor Training Progress

Poll the job status from the notebook:

```python
from sagemaker.core.resources import TrainingJob

response = TrainingJob.get(training_job_name=TRAINING_JOB_NAME)
print(f"Status: {response.training_job_status}")
print(f"Secondary: {response.secondary_status}")
```

Or navigate to **Jobs** → **Training** in SageMaker Studio and open the **JumpStart training** tab:

![Studio Training Jobs](/static/images/lab-1-sft/studio-jobs-training.png)

Click the job to open its detail page, which shows the base model, both datasets, the customization technique, and a **Performance** tab with:

- **Training Loss** — decreasing loss indicates the model is learning
- **Validation Loss** — monitors overfitting on held-out data

### Training Metrics with MLflow

All training metrics are automatically logged to the associated default MLflow app. Click **View full metrics in MLflow** on the training job detail page to explore the full run.

---

## Option 2: Customize with UI

You can also run serverless customization directly from the SageMaker Studio UI without writing code.

In the **Models** section, find your model and choose **Customize** → **Customize with UI**:

![Studio Models Customize](/static/images/lab-1-sft/studio-models-customize.png)

Search for `NVIDIA` to find the Nemotron 3 family and select the model you want (`NVIDIA-Nemotron-3-Nano-30B-*` or `NVIDIA-Nemotron-3-Super-120B-*`).

The customization wizard lets you:

- Select the customization technique (Supervised Fine-Tuning)
- Upload a new dataset or pick an existing one
- Configure hyperparameters (batch size, learning rate, epochs)
- Set the output S3 location

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach for more flexibility and reproducibility. The UI is great for quick experiments and prototyping.
:::
