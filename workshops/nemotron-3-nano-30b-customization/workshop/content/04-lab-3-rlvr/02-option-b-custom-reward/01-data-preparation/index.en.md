---
title: "Data Preparation"
weight: 1
---

::alert[📒 Open the notebook **`lab-3a-custom-reward-function-rlvr/1-prepare-data.ipynb`**]

Select `Python 3 (ipykernel)` for the notebook kernel.

:image[Kernel Selection]{src="/static/images/lab-3-rlvr/kernel-selection.png" width=400 height=300}

### Dataset overview

This option reuses the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) math dataset from Option A, but prepares it for an RLVR job that uses a **custom Python reward function**. The key difference is that each record carries a stable `id` and a top-level `reference_answer` field — both are consumed by the custom reward function at training time to score the model's response.

### Load the dataset

The notebook streams a small sample from GSM8K and shuffles it. We keep the sample count small for workshop runtime and cost — increase `sample_count` for larger experiments:

```python
import datasets
import pandas as pd

sample_count = 180
dataset_stream = datasets.load_dataset("openai/gsm8k", "main", split="train", streaming=True)
dataset = dataset_stream.take(sample_count)
dataset = datasets.Dataset.from_generator(lambda: dataset, features=dataset_stream.features)
```

Each raw GSM8K sample contains a `question` and an `answer` with step-by-step reasoning ending in `#### <number>`.

### Transform to custom-reward RLVR format

The custom reward function receives `reference_answer` at runtime and scores the last assistant message in the `messages` list. The training dataset therefore keeps the normal RLVR fields and **also** stores a stable `id` and a top-level `reference_answer` for the evaluator:

```python
import re


def extract_answer(answer_text):
    match = re.search(r"####\s*(.+)", answer_text)
    return match.group(1).strip().replace(",", "") if match else ""


def prepare_custom_reward_sample(sample, index):
    ground_truth = extract_answer(sample["answer"])
    record_id = f"gsm8k-train-{index:06d}"
    return {
        "id": record_id,
        "data_source": "openai/gsm8k",
        "prompt": [
            {
                "role": "user",
                "content": f"{sample['question']} Let's think step by step and output the final answer after \"####\".",
            }
        ],
        "ability": "math",
        "reward_model": {
            "ground_truth": ground_truth,
            "style": "custom",
        },
        "reference_answer": ground_truth,
        "extra_info": {
            "answer": sample["answer"],
            "question": sample["question"],
            "index": index,
            "split": "train",
        },
    }
```

Key fields that differ from Option A:

- **id**: A stable, unique record identifier (`gsm8k-train-000001`). The reward function returns scores keyed by this `id`.
- **reward_model.style**: Set to `"custom"` (Option A used `"rule"`) — the score comes from your registered reward function rather than the built-in rule verifier.
- **reference_answer**: A top-level copy of the correct answer. The reward function reads this field to verify the model's output.

### Split and upload to Amazon S3

The notebook splits the data into train, validation, and test sets and uploads them to S3:

```python
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(records, test_size=0.2, random_state=42)
train_data, test_data = train_test_split(train_data, test_size=0.1, random_state=42)

input_path = f"{default_prefix + '/' if default_prefix else ''}datasets/{project_prefix}"
train_s3_path = f"s3://{bucket_name}/{input_path}/train/train_rlvr_custom_reward.jsonl"
val_s3_path = f"s3://{bucket_name}/{input_path}/val/val_rlvr_custom_reward.jsonl"
test_s3_path = f"s3://{bucket_name}/{input_path}/test/test_rlvr_custom_reward.jsonl"
```

### Register datasets in SageMaker AI Registry

The prepared datasets are registered as **SageMaker AI Datasets** with the RLVR customization technique:

```python
from sagemaker.ai_registry.dataset import CustomizationTechnique, DataSet

dataset_train = DataSet.create(
    name=f"{project_prefix}-train",
    source=train_s3_path,
    customization_technique=CustomizationTechnique.RLVR,
    wait=True,
)

dataset_val = DataSet.create(
    name=f"{project_prefix}-val",
    source=val_s3_path,
    customization_technique=CustomizationTechnique.RLVR,
    wait=True,
)

dataset_test = DataSet.create(
    name=f"{project_prefix}-test",
    source=test_s3_path,
    customization_technique=CustomizationTechnique.RLVR,
    wait=True,
)
```

The `project_prefix` (`gsm8k-custom-reward-rlvr`) is reused across all notebooks in this lab, so dataset names stay consistent.

:::alert{header="Important" type="info"}
SageMaker AI Datasets are registered in the AI Registry and can be reused across multiple training jobs.
:::

---

Once the datasets are created, you're ready to author the custom reward function.
