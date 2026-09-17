---
title: "Fine-Tuning"
weight: 20
---

## What is Serverless Fine-Tuning on SageMaker?

Normally, fine-tuning a 30B-parameter model means sourcing the right GPU instances, configuring distributed training, managing checkpoints, and tearing everything down when you're done. That's a lot of infrastructure for what is really a data and modeling problem.

SageMaker serverless customization takes all of that off your plate. You tell it what base model to use, which dataset to train on, and which technique to apply. SageMaker provisions the right GPU instances automatically (P5, P4de, P4d, or G5 depending on model size), runs the job using pre-optimized training recipes, streams metrics in real time, and cleans up the compute when training finishes. You pay only for what you use.

The result is that you go from submitting a job to a registered, deployable model in roughly the time it takes to run the training itself, with no cluster management in between.

**Supported techniques:**

| Technique | What it does |
|-----------|-------------|
| **SFT** (Supervised Fine-Tuning) | Trains on labeled prompt/completion pairs. Use this when you have examples of the exact output you want. |
| **DPO** (Direct Preference Optimization) | Trains on preference pairs (chosen vs. rejected). Use this to shape tone, style, or safety behavior. |
| **RLVR** (RL with Verifiable Rewards) | Uses a code-based reward function to score outputs. Use this when correctness can be checked programmatically (math, code, structured JSON). |
| **RLAIF** (RL with AI Feedback) | Uses an LLM as the reward signal. Use this when quality judgement requires natural language reasoning. |

This lab uses **SFT** because the ContractNLI dataset provides the exact JSON output we want the model to produce for each contract.

:::alert{header="Lineage is captured automatically" type="info"}
Every training job records its inputs (dataset name and version), outputs (model package version), and hyperparameters in SageMaker's lineage graph. You can trace any deployed model back to the exact dataset and job that produced it, without keeping notes.
:::

---

## About Nemotron

Nemotron is NVIDIA's family of open-weight language models designed for reasoning and agentic tasks. The models ship with open weights, training data, and recipes, making them practical to fine-tune and deploy in production.

There are three models in the family, spanning two generations:

| Model | Generation | Total params | Active per token | Notes |
|-------|-----------|-------------|-----------------|-------|
| **Nemotron 3 Nano 30B** | 3.0 | 30B | ~3B | Used in this lab |
| **Nemotron 3.5 Lightning 30B** | 3.5 | 30B | ~3B | Successor to Nano; distilled from Ultra |
| **Nemotron 3 Super 120B** | 3.0 | 120B | ~12B | Mid-tier; larger capacity |

### Nano vs Lightning: same size, different generation

Nemotron 3.5 Lightning was formerly called Nemotron 3.5 Nano. NVIDIA renamed it at GA to lead with speed rather than size. It's not a separate model sitting alongside Nano. It's the next generation of it.

The key differences between the two generations:

| | Nemotron 3 Nano (this lab) | Nemotron 3.5 Lightning |
|-|---------------------------|----------------------|
| Training | 25T tokens, SFT, GRPO RL | Distilled from Nemotron 3 Ultra with multi-token prediction (MTP) |
| Throughput | Baseline | Up to 4x higher; 30% faster task completion |
| Agentic perf | Good | Optimized for high-volume multi-model agentic systems |
| Customization | LoRA, full fine-tuning | LoRA, NeMo RL, NVFP4 quantized variants |
| Deployment | Cloud | Edge to datacenter (DGX Spark, Jetson, RTX) |

This lab uses **Nemotron 3 Nano 30B** because it's the generation the ContractNLI fine-tuning recipe was built and validated on. The same approach applies to Lightning with a model ID swap in `config.py`.

### What "30B-A3B" means

The name tells you the shape of the model: 30 billion total parameters, with roughly 3 billion active on any given token. That's possible because of the **Mixture-of-Experts (MoE)** architecture.

Instead of a single dense feed-forward block at each layer, MoE layers hold many "expert" sub-networks and a router that picks which ones to activate per token. Nemotron 3 Nano has 128 routed experts plus one shared expert per MoE layer, activating 6 per token. The rest stay dormant. You get the knowledge capacity of a 30B model at something closer to the inference cost of a 3B one.

The architecture combines:
- **23 Mamba-2 and MoE layers** for efficient long-context processing
- **6 Attention layers** for global context

### Reasoning model behavior

Both Nano and Lightning are reasoning models. Before answering, they generate an internal chain of thought inside a `<think>...</think>` block. This is what makes them strong at multi-step tasks like contract review, where the right answer depends on reading the whole document carefully.

You'll see how this affects training data in the data preparation section, and why completions start with `<think>\n</think>\n`.

### Why fine-tune for this task?

The base model already knows how to reason and follow structured instructions. What it doesn't know is the specific 17-item ContractNLI checklist, the exact JSON output format, or the legal reasoning patterns that distinguish `Contradiction` from `NotMentioned` for NDA clauses. Fine-tuning teaches it exactly those things, while keeping all the general capabilities intact.

---

## Understanding LoRA

LoRA (Low-Rank Adaptation) fine-tunes a model without touching most of its weights. Instead of updating all parameters, it injects small adapter matrices at the attention layers and trains only those. The rest of the model stays frozen.

The practical result: you're training a fraction of the parameters (10-100x fewer), using far less GPU memory, and ending up with a compact adapter you can swap in and out without keeping a separate full copy of the model per task.

![LoRA Architecture](/static/images/lab-1-sft/lora_animated.gif)

:image[LoRA vs Full Fine-Tuning]{src="/static/images/general/peft_lora_concept.jpeg" height=320}

### Key Hyperparameters

| Parameter | Description | Typical Values |
| -------------- | ------------------------ | -------------- |
| `lora_rank` | Rank of adapter matrices (higher rank = more capacity, more parameters) | 4-64 |
| `lora_alpha` | Scaling factor for the adapter output | 2-4x the rank |
| `lora_dropout` | Regularization applied to the adapter layers | 0.05-0.1 |

---

## Option 1: Customize with Code

::alert[Open the notebook **`lab-1-supervised-fine-tuning/2-fine-tune-llm.ipynb`**]

### Prerequisites

Retrieve the datasets created in the previous step. The base model is defined once in `config.py` and imported here, so every notebook in the lab stays in sync:

```python
from sagemaker.ai_registry.dataset import DataSet
from config import BASE_MODEL_ID

base_model_id = BASE_MODEL_ID # "huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16"

training_dataset = DataSet.get(name="contractnli-nda-review-train")
val_dataset = DataSet.get(name="contractnli-nda-review-val")
```

:::alert{header="Note" type="info"}
Want to try a different model? The notebook includes a cell that lists all SageMaker JumpStart models supporting customization. To switch, just update `BASE_MODEL_ID` in `config.py` and every notebook in this lab picks up the change automatically. Not every model supports every technique, so a "No recipes found" error means the model doesn't support SFT.
:::

### Create Model Package Group

Model Package Groups organize your fine-tuned model versions. The name is derived from the base model ID with a `-contractnli-sft-mpg` suffix, hash-truncated to SageMaker's 63-character limit when needed. Notebooks 2, 3, and 4 all derive it the same way:

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

:::alert{header="About `ModelPackageGroup`" type="info"}
`sagemaker.core.resources.ModelPackageGroup` is a container in the [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html) that holds successive versions of a model, one per fine-tuning run. It records the lineage for each version (which dataset, job, and hyperparameters produced it) and provides a stable name that Labs 3 and 4 can use to look up the latest model without hardcoding ARNs. [API documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-model-package-groups.html)
:::

### Customization techniques available

SageMaker AI serverless customization supports three techniques for NVIDIA Nemotron 3 Nano 30B and Super 120B. This lab uses SFT because the ContractNLI training records are exact labeled examples: the right output for each contract is known.

:image[Customization techniques]{src="/static/images/general/customization_techniques.jpg" height=240}

| Technique | When to use |
|-----------|-------------|
| **SFT** | You have labeled prompt/completion pairs and want the model to produce specific outputs |
| **RLVR** | You have a verifiable reward function (e.g., a code execution result or a math check) |
| **RLAIF** | You have preference data and use an AI model as the reward signal |

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

:::alert{header="About `SFTTrainer`" type="info"}
`sagemaker.train.sft_trainer.SFTTrainer` is the SageMaker AI SDK class for serverless supervised fine-tuning. It handles recipe discovery (mapping the base model to a supported fine-tuning recipe), job submission, and registration of the resulting model package. You describe *what* to fine-tune; the SDK and SageMaker handle compute provisioning, distributed training, and artifact storage. `accept_eula=True` acknowledges the base model's end-user license agreement, required by JumpStart before any training begins. [API documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-customization-sft.html)
:::

:::alert{header="About `TrainingType`" type="info"}
`sagemaker.train.common.TrainingType` is an enum with two values: `LORA` and `FULL`. `LORA` trains only small adapter matrices (far cheaper and faster). `FULL` updates all model parameters (not supported for the largest models at this time). This lab uses `LORA`.
:::

### View Default Hyperparameters

```python
from rich.pretty import pprint

print("Default Finetuning options:")
pprint(trainer.hyperparameters.to_dict())
```

### Override Hyperparameters

These are the six values that matter most for this run. The warning below explains why `max_epochs = 10` isn't arbitrary:

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

Ten is not arbitrary. The output format and the label distribution are learned early, but `evidence_f1` (whether you cited the clause a reviewer would actually check) keeps improving well after accuracy flattens, and it is the metric this task is judged on. Shorter runs leave it short of the frontier baseline measured in the next section.
:::

### Submit the Training Job

```python
training_job = trainer.train(wait=False)

TRAINING_JOB_NAME = training_job.training_job_name
pprint(training_job)
```

:::alert{header="Important" type="warning"}
The serverless fine-tuning job takes **20-25 minutes** for the 10 epochs configured above. SageMaker AI provisions the compute for you, and that share is fixed, so the wall-clock does not scale with epochs the way you might expect. Halving the epochs does not halve the time. With a larger dataset the training share grows while the start-up cost stays roughly where it is.
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

:::alert{header="The evaluation notebook requires a completed training job" type="warning"}
Lab 3 scores the fine-tuned model directly, so you need to wait for this job to reach `Completed` before running it. The job takes 20-25 minutes. Submit it with `wait=False`, then use the polling cell or the AWS Console (SageMaker AI > Training > Training jobs) to watch progress. Once it completes, continue to Lab 3.
:::

---
