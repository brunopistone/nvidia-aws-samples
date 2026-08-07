---
title: "Lab 1️⃣: Supervised Fine-Tuning (SFT)"
weight: 20
---

In this lab, you will fine-tune **NVIDIA Nemotron 3 Nano 30B-A3B** using Serverless Supervised Fine-Tuning on Amazon SageMaker AI.

SFT is the most straightforward customization technique — you provide prompt/completion pairs and the model learns to produce the desired outputs. It's ideal for teaching a model a new output format, a new style, or a structured reasoning behaviour.

:image[Supervised Fine-Tuning concept]{src="/static/images/lab-1-sft/sft_concept.jpeg" height=384}

## What you will learn

1. **Prepare your dataset** - Format data for SFT and create SageMaker AI Datasets
2. **Run a serverless fine-tuning job** - Use the `SFTTrainer` API with LoRA
3. **Evaluate your model** - Use LLM-as-a-Judge with task-specific custom metrics
4. **Deploy your model** - Serve the customized model (covered in [Lab 2](/03-lab-inference/))

## Model and Dataset

| Component      | Details                                                                                                                |
| -------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Base Model** | NVIDIA Nemotron 3 Nano 30B-A3B (`huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16`)                            |
| **Dataset**    | [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)              |
| **Technique**  | Supervised Fine-Tuning with LoRA                                                                                       |
| **Use Case**   | Multilingual chain-of-thought: reason in a target non-English language, answer in English                               |

## The task: reason in one language, answer in another

The **Multilingual-Thinking** dataset teaches the model a very specific behaviour, driven entirely by the system prompt:

```text
You are an AI assistant that thinks in {language} but responds in English.

IMPORTANT: Follow this exact format for every response:
1. First, write your reasoning and thoughts inside <think>...</think> tags
2. Then, provide your final answer in English

Always think through the problem in {language}, then translate your conclusion to English for the final response.
```

`{language}` is one of **Spanish**, **French**, **Italian** or **German**.

This makes a great fine-tuning demonstration because the target behaviour is:

- **Easy to verify** — you can see at a glance whether the reasoning is in the right language and the answer is in English
- **Structurally strict** — the model must emit exactly one well-formed `<think>...</think>` block
- **Not something the base model does reliably** — so the improvement from fine-tuning is visible

## Why fine-tune for this?

- Teach the model to **follow a strict output contract** (`<think>` reasoning, then the answer)
- Make **reasoning language** controllable through the system prompt
- Keep the final answer in a **consistent language** for downstream consumers

## Lab Structure

| Section                                 | Description                                                 | Duration   |
| --------------------------------------- | ----------------------------------------------------------- | ---------- |
| [Data Preparation](01-data-preparation) | Prepare and upload the dataset to S3, create SageMaker AI Datasets | ~10 min    |
| [Fine-Tuning](02-fine-tuning)           | Launch the serverless SFT job with LoRA                     | ~20-30 min |
| [Evaluation](03-evaluation)             | Evaluate with LLM-as-a-Judge and custom metrics             | ~20-25 min |

:::alert{header="Important" type="warning"}
This workshop is designed to demonstrate serverless model customization, not to produce a production-grade model. The Multilingual-Thinking dataset is small (about 1,000 examples), which keeps the training job short. You can adapt this codebase to a larger dataset as needed.
:::
