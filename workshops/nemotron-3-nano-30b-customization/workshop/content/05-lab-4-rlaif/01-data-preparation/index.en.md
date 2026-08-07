---
title: "Data Preparation"
weight: 1
---

## About the Dataset

**[HumanLLMs/Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset)** is a dataset designed to train models to generate more natural, human-like conversational responses. It contains prompts with human-written responses that exhibit natural language patterns, emotional intelligence, and conversational flow.

---

## Prepare Data with Code

::alert[📒 Open the notebook **`lab-4-reinforcement-learning-from-ai-feedback/1-prepare-data.ipynb`**]

Select `Python 3 (ipykernel)` for the notebook kernel.

### Dataset structure

The dataset contains conversational prompts with human-like responses:

```python
dataset = load_dataset(
    "HumanLLMs/Human-Like-DPO-Dataset",
    split="train"
)
```

### Format your dataset for RLAIF training

For RLAIF, we need to follow the [VERL post-training format](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) required by SageMaker's RLAIFTrainer:

```python
def prepare_dataset_sm_rlaif(sample):
    return {
        "data_source": "HumanLLMs/Human-Like-DPO-Dataset",
        "prompt": [
            {
                "role": "user",
                "content": sample.get("prompt")
            }
        ],
        "ability": "human-like",
        "reward_model": {
            "ground_truth": "",
            "style": "llmj",
        }
    }
```

This creates training examples where:

- **data_source**: Dataset identifier
- **prompt**: Conversational messages in chat format
- **ability**: The capability being trained
- **reward_model**: Specifies the evaluation style (`llmj` for LLM-as-a-Judge)

:::alert{header="Important" type="info"}
The `ground_truth` field is intentionally left empty in the training dataset. During RLAIF training, the model generates its own candidate responses which are scored by the AI judge — no reference answers are needed. The human-written `chosen` responses from the original dataset are only used in the separate evaluation dataset, where they serve as ground truth for the LLM-as-a-Judge to compare against.
:::

Example formatted row:

```json
{
  "data_source": "HumanLLMs/Human-Like-DPO-Dataset",
  "prompt": [
    {
      "role": "user",
      "content": "How do you stay motivated when working on long-term projects?"
    }
  ],
  "ability": "human-like",
  "reward_model": {
    "ground_truth": "",
    "style": "llmj"
  }
}
```

### Format your dataset for evaluation

For evaluation with LLM-as-a-Judge, we use a different format:

```python
def prepare_dataset_sm_eval(sample):
    return {
        "query": sample.get("prompt"),
        "response": sample.get("chosen")
    }
```

### Upload to Amazon S3

The notebook uploads the prepared datasets to S3:

```python
train_data_uri = f"s3://{bucket_name}/{project_prefix}/{project_prefix}_train.jsonl"
eval_data_uri = f"s3://{bucket_name}/{project_prefix}/{project_prefix}_eval.jsonl"
```

### Create SageMaker AI Datasets

Register the datasets as reusable assets in the SageMaker AI registry:

```python
from sagemaker.ai_registry.dataset import DataSet

train_dataset = DataSet.create(
    name=f"{project_prefix}-train",
    source=train_data_uri,
    sagemaker_session=sess,
    wait=True,
)

eval_dataset = DataSet.create(
    name=f"{project_prefix}-eval",
    source=eval_data_uri,
    sagemaker_session=sess,
    wait=True,
)
```

This makes the datasets available in the SageMaker AI registry for use in the subsequent training and evaluation notebooks.

:::alert{header="Important" type="info"}
RLAIF datasets use the VERL post-training format with nested structures for prompts and reward model configuration. The `ground_truth` field is left empty for training — the AI judge scores model-generated responses without reference answers. Ground truth responses are only used in the evaluation dataset.
:::

---

Once the datasets are created, you're ready to configure the reward model.
