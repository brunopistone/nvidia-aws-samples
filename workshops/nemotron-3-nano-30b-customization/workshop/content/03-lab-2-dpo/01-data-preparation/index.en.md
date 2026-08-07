---
title: "Data Preparation"
weight: 1
---

::alert[📒 Open the notebook **`lab-2-direct-preference-optimization-DPO/1-dpo-prepare-data.ipynb`**]

Select `Python 3 (ipykernel)` for the notebook kernel.

:image[Kernel Selection]{src="/static/images/lab-2-dpo/kernel-selection.png" width=400 height=300}

## About the Dataset

**[HumanLLMs/Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset)** is a dataset that was created as part of research aimed at improving conversational fluency and engagement in large language models. It is suitable for formats like Direct Preference Optimization (DPO) to guide models toward generating more human-like responses.

## Dataset structure

The original dataset contains conversational prompts with human-like and AI-style response pairs:

```python
dataset = (
    load_dataset(
        "HumanLLMs/Human-Like-DPO-Dataset",
        split="train",
        streaming=True,
    )
    .take(3000)
    .shuffle(buffer_size=1000)
)
```

## Format your dataset for training and validation

We format the dataset for Serverless DPO using a simple prompt, chosen and rejected structure:

```python
def prepare_dataset_sm_dpo_train_val(sample):
    try:
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"]
        }
    except Exception as e:
        print(f"Error: {e}")

        raise e
```

This creates training examples where:
- **prompt**: Natural, engaging questions that reflect everyday human dialogue.
- **chosen**: A natural, conversational answer generated to mimic human interaction.
- **rejected**: A structured, professional answer reflecting traditional AI responses.

Example formatted row:

```json
{
  "prompt": "What's one thing you're really looking forward to doing this summer?",
  "chosen": "You know, I'm really excited to just relax and enjoy the longer days! There's something about summer that just feels more laid back, you know? But if I had to pick one thing, I'm really looking forward to having a BBQ with friends and family. There's nothing like grilled burgers, cold drinks, and good company to make the summer vibes feel real 😊. How about you, do you have any fun plans for the summer?",
  "rejected": "Good morning. While personal preferences may vary, research suggests that many individuals appreciate the opportunity to sleep in and enjoy a leisurely morning on Saturdays. The absence of a strict weekday schedule can provide a sense of relief and freedom, allowing individuals to prioritize their own needs and pursuits. Additionally, the extra time can be utilized for relaxation, self-care, or engaging in enjoyable activities, which can contribute to an overall sense of well-being and rejuvenation."
}
```

## Format your dataset for testing

For evaluation with LLM-as-a-Judge, we use a different format:

```python
def prepare_dataset_sm_dpo_test(sample):
    try:
        return {
            "query": sample["prompt"],
            "response": sample["chosen"]
        }
    except Exception as e:
        print(f"Error: {e}")

        raise e
```

## Upload to Amazon S3

The notebook uploads the prepared datasets to S3:

```python
train_dataset_s3_path = f"s3://{bucket_name}/{input_path}/train/humanlike_dpo_train.jsonl"
val_dataset_s3_path = f"s3://{bucket_name}/{input_path}/val/humanlike_dpo_val.jsonl"
test_dataset_s3_path = f"s3://{bucket_name}/{input_path}/test/humanlike_dpo_test.jsonl"
```

## Create SageMaker AI Datasets

The key step for serverless customization is creating **SageMaker AI Datasets**:

```python
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique

dataset_train = DataSet.create(
    name="humanlike-dpo-train",
    source=train_dataset_s3_path,
    customization_technique=CustomizationTechnique.DPO,
    wait=True,
)

dataset_val = DataSet.create(
    name="humanlike-dpo-val",
    source=val_dataset_s3_path,
    customization_technique=CustomizationTechnique.DPO,
    wait=True,
)

dataset_test = DataSet.create(
    name="humanlike-dpo-test",
    source=test_dataset_s3_path,
    wait=True,
)
```

The `CustomizationTechnique.DPO` parameter tells SageMaker AI to validate the dataset format for Direct Preference Optimization (DPO).

:::alert{header="Important" type="info"}
SageMaker AI Datasets are registered in the AI Registry and can be reused across multiple fine-tuning jobs.
:::

## Alternative: Create Datasets with UI

You can also create datasets directly from the SageMaker Studio UI. Navigate to **Assets** → **Datasets** and click **Upload Dataset**:

![Studio Datasets Upload](/static/images/lab-2-dpo/studio-dpo-datasets-upload.png)

The UI shows the required data format for each customization technique and allows you to upload JSONL files or specify an S3 URI.

---

Once the datasets are created, you're ready to start the fine-tuning job.
