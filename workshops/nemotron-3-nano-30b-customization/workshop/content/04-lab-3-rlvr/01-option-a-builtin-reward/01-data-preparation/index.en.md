---
title: "Data Preparation"
weight: 1
---

## Option 1: Prepare Data with Code

::alert[📒 Open the notebook **`lab-3-reinforcement-learning-from-verifiable-rewards/1-prepare-data.ipynb`**]

Select `Python 3 (ipykernel)` for the notebook kernel.

:image[Kernel Selection]{src="/static/images/lab-3-rlvr/kernel-selection.png" width=400 height=300}

### Dataset overview

RLVR requires a dataset with verifiably correct answers — problems where the model's output can be automatically checked against a ground truth. We use the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) math dataset, which contains grade-school math word problems with numerical answers.

### Load the dataset

The notebook streams 180 samples from the GSM8K dataset on HuggingFace and shuffles them:

```python
import datasets
from datasets import load_dataset

dataset = (
    load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    .take(180)
    .shuffle(buffer_size=180)
)

dataset = datasets.Dataset.from_generator(lambda: dataset, features=dataset.features)
```

Each raw GSM8K sample contains a `question` and an `answer` with step-by-step reasoning ending in `#### <number>`.

### Transform to RLVR format

The notebook transforms each sample into the RLVR format required by SageMaker AI:

```python
import re


def extract_answer(answer_text):
    """Extract the final numerical answer after ####"""
    match = re.search(r"####\s*(.+)", answer_text)
    return match.group(1).strip().replace(",", "") if match else ""


def prepare_rlvr_sample(sample, index):
    """Convert a single GSM8K sample into the RLVR training format.

    The `split` field in extra_info is filled in later (after the train/val/test
    split) so each record is labeled with the split it actually belongs to.
    """
    ground_truth = extract_answer(sample["answer"])
    yield {
        "data_source": "openai/gsm8k",
        "prompt": [
            {
                "content": f"{sample['question']} Let's think step by step and output the final answer after \"####\".",
                "role": "user",
            }
        ],
        "ability": "math",
        "reward_model": {"ground_truth": ground_truth, "style": "rule"},
        "extra_info": {
            "answer": sample["answer"],
            "index": index,
            "question": sample["question"],
        },
    }
```

This produces records in the following structure:

```json
{
  "data_source": "openai/gsm8k",
  "prompt": [
    {
      "content": "Natalia sold clips to 48 of her friends in April... Let's think step by step and output the final answer after \"####\".",
      "role": "user"
    }
  ],
  "ability": "math",
  "reward_model": {
    "ground_truth": "72",
    "style": "rule"
  },
  "extra_info": { ... }
}
```

Key fields:

- **prompt**: A chat-formatted message with the math problem, instructing the model to think step by step and output the final answer after `####`.
- **reward_model.ground_truth**: The correct numerical answer used to verify the model's output during training.
- **reward_model.style**: Set to `"rule"` — a rule-based reward function checks if the model's answer matches the ground truth.

Unlike SFT where you provide the ideal completion, RLVR only needs the question and the correct answer. The model learns to generate its own reasoning path through trial and reward.

### Split the dataset

The notebook splits the transformed data into train, validation, and test sets:

```python
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(records, test_size=0.2, random_state=42)
train_data, test_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Label each record with the split it actually landed in, so extra_info["split"]
# matches the JSONL file it's written to.
for split_name, split_records in [("train", train_data), ("val", val_data), ("test", test_data)]:
    for record in split_records:
        record["extra_info"]["split"] = split_name
```

### Upload to Amazon S3

The notebook uploads the prepared datasets to S3:

```python
train_s3_path = f"s3://{bucket_name}/{input_path}/train/train_rlvr.jsonl"
val_s3_path = f"s3://{bucket_name}/{input_path}/val/val_rlvr.jsonl"
test_s3_path = f"s3://{bucket_name}/{input_path}/test/test_rlvr.jsonl"
```

### Register datasets in SageMaker AI Registry

The key step for serverless customization is registering **SageMaker AI Datasets** in the AI Registry:

```python
from sagemaker.ai_registry.dataset import DataSet

dataset_train = DataSet.create(
    name="rlvr-train",
    source=train_s3_path,
    wait=True,
)

dataset_val = DataSet.create(
    name="rlvr-val",
    source=val_s3_path,
    wait=True,
)

dataset_test = DataSet.create(
    name="rlvr-test",
    source=test_s3_path,
    wait=True,
)
```

:::alert{header="Important" type="info"}
SageMaker AI Datasets are registered in the AI Registry and can be reused across multiple training jobs.
:::

## Option 2: Create Datasets with UI

If you have an already formatted dataset in JSONL format, you can also create datasets directly from the SageMaker Studio UI. Navigate to **Assets** → **Datasets** and click **Upload Dataset**:

![Studio Datasets Upload](/static/images/lab-3-rlvr/studio-datasets-upload.png)

The UI shows the required data format for each customization technique and allows you to upload JSONL files or specify an S3 URI.

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach for more flexibility and reproducibility. The UI is great for quick experiments and prototyping.
:::

---

Once the datasets are created, you're ready to start the RLVR training job.
