# SFT with Serverless Model Customization on Amazon SageMaker AI

An end-to-end, notebook-based lab that fine-tunes **NVIDIA Nemotron 3 Nano 30B-A3B** with
**LoRA Supervised Fine-Tuning (SFT)** using SageMaker AI serverless model customization,
then evaluates and deploys the resulting model.

The model is fine-tuned on the
[HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
dataset so that it **reasons inside `<think>...</think>` tags in a target non-English language**
(Spanish, French, Italian, or German — selected via the system prompt) and then **gives its final
answer in English**.

---

## Pipeline

```
1-prepare-data  ->  2-fine-tune-llm  ->  3-evaluation  ->  4-deployment       (real-time endpoint)
                                                       \->  4a-deployment-bedrock (Amazon Bedrock)
```

| Notebook                      | Purpose                                                                                                                                                                                                                             |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1-prepare-data.ipynb`        | Load `Multilingual-Thinking`, split 70/20/10 into train/val/test, format each split (prompt/completion for SFT; GenQA `query`/`response` for the eval test set), upload to S3, and register them as SageMaker AI Registry datasets. |
| `2-fine-tune-llm.ipynb`       | Launch a serverless **LoRA SFT** job with `SFTTrainer`, register a Model Package Group, and submit the training job.                                                                                                                |
| `3-evaluation.ipynb`          | Run an **LLM-as-a-Judge** evaluation (Amazon Nova Pro) with built-in and task-specific custom metrics, download the results from S3, and visualize them.                                                                            |
| `4-deployment.ipynb`          | Deploy the merged model to a **SageMaker real-time endpoint** using the **vLLM** Deep Learning Container plus a custom `nano_v3` reasoning parser (on `ml.g5.12xlarge`).                                                            |
| `4a-deployment-bedrock.ipynb` | Deploy the fine-tuned model to **Amazon Bedrock** via Custom Model Import.                                                                                                                                                          |

---

## The reasoning task

The system prompt instructs the model to think in a chosen language and answer in English:

```text
You are an AI assistant that thinks in {language} but responds in English.

IMPORTANT: Follow this exact format for every response:
1. First, write your reasoning and thoughts inside <think>...</think> tags
2. Then, provide your final answer in English

Always think through the problem in {language}, then translate your conclusion to English.
```

`{language}` is one of `Spanish`, `French`, `Italian`, `German`.

---

## Prerequisites

- An AWS account with **Amazon SageMaker AI** access (this lab uses **`us-east-1`**).
- A **SageMaker execution role**. For the evaluation step, the role's **trust policy must include
  `bedrock.amazonaws.com`** (LLM-as-a-Judge runs on Amazon Bedrock). See
  [Bedrock permissions setup](https://docs.aws.amazon.com/bedrock/latest/userguide/judge-service-roles.html).
- **Amazon Bedrock** model access for the judge model (`amazon.nova-pro-v1:0`) and, for `4a`, for
  Custom Model Import.
- Access to the base model in **SageMaker JumpStart** (the training job passes `accept_eula=True`).
- Service quota for a GPU endpoint instance (**`ml.g5.12xlarge`**, 4x NVIDIA A10G) for real-time deployment.
- **Python 3.12**.

---

## Getting started

```bash
# 1. Clone
git clone <your-repo-url>
cd nvidia-nemotron-3-nano-sft

# 2. (Recommended) create an environment
python3.12 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

The base model is configured once in [`config.py`](config.py) and picked up by every notebook:

```python
BASE_MODEL_ID = "huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16"
```

To try a different customizable JumpStart model, change `BASE_MODEL_ID` — notebook 2 includes a cell
that lists models supporting customization.

Then run the notebooks in order (`1` -> `2` -> `3` -> `4`/`4a`).

---

## Default fine-tuning hyperparameters

Set in `2-fine-tune-llm.ipynb` (LoRA SFT):

| Hyperparameter      | Value  |
| ------------------- | ------ |
| `learning_rate`     | `1e-4` |
| `global_batch_size` | `128`  |
| `max_epochs`        | `4`    |
| `lora_rank`         | `16`   |
| `lora_alpha`        | `32`   |
| `warmup_steps`      | `3`    |
| `weight_decay`      | `0.01` |

> Note: serverless model customization enforces a minimum `global_batch_size` of **128**.

---

## Evaluation metrics

`3-evaluation.ipynb` scores the model on the held-out test split (0–1 scale):

- **Built-in:** Correctness, Completeness, Faithfulness, Coherence.
- **Custom (task-specific):**
  - `ReasoningLanguageAdherence` — the `<think>` reasoning is in the target non-English language.
  - `EnglishFinalAnswer` — the final answer (after `</think>`) is in fluent English.
  - `ThinkTagStructure` — exactly one well-formed `<think>...</think>` block followed by an answer.
  - `ReasoningQuality` — reasoning is on-topic, coherent, and supports the answer.

---

## Deployment notes

**Real-time endpoint (`4-deployment.ipynb`)** uses the SageMaker vLLM DLC with a custom `nano_v3`
reasoning parser so the chain-of-thought is returned **separately from the final answer**. With this
container (vLLM 0.22.x), the OpenAI-compatible response places the chain-of-thought in the
**`reasoning`** field and the answer in **`content`**:

```python
msg = json.loads(resp["Body"].read())["choices"][0]["message"]
print(msg["reasoning"])  # chain-of-thought
print(msg["content"])    # final English answer
```

When streaming, read the delta field defensively, since the name varies by vLLM version:

```python
reasoning = delta.get("reasoning") or delta.get("reasoning_content")
```

---

## Cleanup

To avoid ongoing charges, delete the real-time endpoint and its resources when you are done — the
final cells of `4-deployment.ipynb` remove the inference component, model, endpoint, and endpoint
config. Serverless training and evaluation jobs do not incur idle costs, but a running endpoint does.

---

## Security

Before publishing this repository:

- **Clear notebook outputs.** Executed cells may embed your AWS account ID, role ARNs, and S3 bucket
  names. Run `jupyter nbconvert --clear-output --inplace *.ipynb` (or use
  [`nbstripout`](https://github.com/kynan/nbstripout)) before committing.
- **Do not hardcode credentials or profiles.** Prefer the default credential chain over a hardcoded
  `AWS_PROFILE`.
- Treat account IDs, ARNs, and bucket names as sensitive and scrub them from committed cells.

---

## Repository structure

```
.
├── 1-prepare-data.ipynb          # Data preparation
├── 2-fine-tune-llm.ipynb         # Serverless LoRA SFT
├── 3-evaluation.ipynb            # LLM-as-a-Judge evaluation
├── 4-deployment.ipynb            # SageMaker real-time endpoint (vLLM)
├── 4a-deployment-bedrock.ipynb   # Amazon Bedrock Custom Model Import
├── config.py                     # BASE_MODEL_ID (shared across notebooks)
├── requirements.txt              # Python dependencies
└── README.md
```

## License

Specify a license for this project (for AWS samples, `MIT-0` is common). Add a `LICENSE` file to the repository root.
