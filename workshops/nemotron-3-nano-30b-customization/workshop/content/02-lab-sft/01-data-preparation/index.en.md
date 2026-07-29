---
title: "Data Preparation"
weight: 1
---

## About the Dataset

**[HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)** is a small reasoning dataset (~1,000 examples) where each conversation contains a chain-of-thought written in a **non-English language** (Spanish, French, Italian or German) and a **final answer in English**.

Each row contains:

- `messages` — the conversation (`system`, `user`, `assistant`); the assistant turn carries a separate `thinking` field
- `reasoning_language` — the target language of the reasoning for that example

---

## Option 1: Prepare Data with Code

::alert[📒 Open the notebook **`code/1-prepare-data.ipynb`**]

Select `Python 3 (ipykernel)` for the notebook kernel.

:image[Kernel Selection]{src="/static/images/lab-1-sft/kernel-selection.png" width=400 height=300}

### Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceH4/Multilingual-Thinking",
    split="train",
)
```

### Split into train / validation / test

The dataset is converted to a pandas DataFrame and split into **70% train / 20% validation / 10% test**:

```python
from sklearn.model_selection import train_test_split

train, val = train_test_split(df, test_size=0.2, random_state=42)
train, test = train_test_split(train, test_size=0.125, random_state=42)

print("Number of train elements: ", len(train))
print("Number of validation elements: ", len(val))
print("Number of test elements: ", len(test))
```

### Format your dataset for training and validation

Serverless SFT expects a **prompt/completion** structure. The notebook rebuilds each row from the original `messages`, generating the system prompt that encodes the target reasoning language and folding the `thinking` field back into `<think>...</think>` tags:

```python
def prepare_dataset_train_val(sample):
    system = None
    prompt = None
    completion = ""

    for el in sample["messages"]:
        if el["role"] == "system":
            system_prompt = """
            You are an AI assistant that thinks in {language} but responds in English.

            IMPORTANT: Follow this exact format for every response:
            1. First, write your reasoning and thoughts inside <think>...</think> tags
            2. Then, provide your final answer in English

            Always think through the problem in {language}, then translate your conclusion to English for the final response.
            """

            system_prompt = system_prompt.format(language=sample["reasoning_language"])
            system = textwrap.dedent(system_prompt).strip()
        elif el["role"] == "user":
            prompt = el["content"]
        else:
            thinking = el.get("thinking")
            if thinking is not None and thinking != "" and thinking != "null":
                completion = f"<think>\n{thinking}\n</think>\n\n"

            completion += el["content"]

    yield {
        "system": system,
        "prompt": prompt,
        "completion": completion,
    }
```

This creates training examples where:

- **system**: the instruction that sets the target reasoning language
- **prompt**: the user question
- **completion**: the reasoning in `<think>` tags (in the target language) followed by the English answer

Example formatted row:

```json
{
  "system": "You are an AI assistant that thinks in Italian but responds in English.\n\nIMPORTANT: Follow this exact format for every response:\n1. First, write your reasoning and thoughts inside <think>...</think> tags\n2. Then, provide your final answer in English\n\nAlways think through the problem in Italian, then translate your conclusion to English for the final response.",
  "prompt": "Can you give me a recipe with chicken and broccoli?",
  "completion": "<think>\nAllora, l'utente chiede una ricetta con pollo e broccoli...\n</think>\n\n Here's a simple chicken and broccoli stir-fry..."
}
```

### Format your dataset for testing

The evaluation job in the next section consumes the **GenQA** format, which uses different column names — `query` for the input and `response` for the reference answer (exposed to the judge as `{{ground_truth}}`):

```python
def prepare_dataset_test(sample):
    ...
    yield {
        "system": system,
        "query": query,
        "response": response,
    }
```

:::alert{header="Why two formats?" type="info"}
Train and validation splits use the SFT `prompt`/`completion` schema. The test split uses the GenQA `query`/`response` schema because it is consumed by the LLM-as-a-Judge evaluation job, not by the trainer.
:::

### Upload to Amazon S3

The notebook writes the three splits as JSONL and uploads them:

```python
input_path = "datasets/serverless-model-customization-sft"

train_dataset_s3_path = f"s3://{bucket_name}/{input_path}/train/dataset.jsonl"
val_dataset_s3_path = f"s3://{bucket_name}/{input_path}/val/dataset.jsonl"
test_dataset_s3_path = f"s3://{bucket_name}/{input_path}/test/dataset.jsonl"
```

### Create SageMaker AI Datasets

The key step for serverless customization is registering the data as **SageMaker AI Datasets**:

```python
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique

dataset_train = DataSet.create(
    name="Multilingual-Thinking-sft-train",
    source=train_dataset_s3_path,
    customization_technique=CustomizationTechnique.SFT,
    wait=True,
)

dataset_val = DataSet.create(
    name="Multilingual-Thinking-sft-val",
    source=val_dataset_s3_path,
    customization_technique=CustomizationTechnique.SFT,
    wait=True,
)

# No customization_technique: the test split is used for evaluation, not training
dataset_test = DataSet.create(
    name="Multilingual-Thinking-sft-test",
    source=test_dataset_s3_path,
    wait=True,
)
```

The `CustomizationTechnique.SFT` parameter tells SageMaker AI to validate the dataset format for Supervised Fine-Tuning.

:::alert{header="Important" type="info"}
SageMaker AI Datasets are registered in the AI Registry and can be reused across multiple fine-tuning jobs. Once created, each one appears as a registered dataset version under **Assets** → **Datasets** in SageMaker Studio, and you can reference it by name in your fine-tuning jobs.
:::

---

## Option 2: Create Datasets with UI

If you already have a formatted dataset in JSONL format, you can also create datasets directly from the SageMaker Studio UI. Navigate to **Assets** → **Datasets** and click **Upload Dataset**:

![Studio Datasets Upload](/static/images/lab-1-sft/studio-datasets-upload.png)

The UI shows the required data format for each customization technique and allows you to upload JSONL files or specify an S3 URI.

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach for more flexibility and reproducibility.
:::

---

Once the datasets are created, you're ready to start the fine-tuning job.
