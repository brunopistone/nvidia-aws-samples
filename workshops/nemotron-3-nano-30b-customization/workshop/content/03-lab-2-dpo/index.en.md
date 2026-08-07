---
title: "Lab 2️⃣: Direct Preference Optimization (DPO)"
weight: 3
---

# Direct Preference Optimization (DPO) Training with SageMaker

This notebook demonstrates how to use the **DPOTrainer** to fine-tune large language models using Direct Preference Optimization (DPO). DPO is a technique that trains models to align with human preferences by learning from preference data without requiring a separate reward model.

## What is DPO?

Direct Preference Optimization (DPO) is a method for training language models to follow human preferences. Unlike traditional RLHF (Reinforcement Learning from Human Feedback), DPO directly optimizes the model using preference pairs without needing a reward model.

:image[Direct Preference Optimization concept]{src="/static/images/lab-2-dpo/dpo_concept.jpeg" height=384}

The key difference from SFT is the data format: instead of single correct answers, DPO uses pairs of preferred and non-preferred responses for each prompt, allowing the model to learn what makes one response better than another.

**Key Benefits:**

- Simpler than RLHF - no reward model required
- More stable training process
- Direct optimization on preference data
- Works with LoRA for efficient fine-tuning

In this lab, you will learn how to fine-tune **Meta Llama 3.2 1B instruct** using Serverless Direct Preference Optimization on Amazon SageMaker AI.

## What you will learn

1. **Prepare your dataset** - Format data for DPO and create SageMaker AI Datasets
2. **Run a serverless fine-tuning job** - Use the DPOTrainer API with LoRA
3. **Evaluate your model** - Use LLM-as-a-Judge with custom metrics
4. **Deploy your model** - Create a SageMaker real-time endpoint

## Model and Dataset

| Component      | Details                                                                                              |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| **Base Model** | Llama 3.2 1B instruct (`meta-textgeneration-llama-3-2-1b-instruct`)                                  |
| **Dataset**    | [HumanLLMs/Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) |
| **Technique**  | Direct Preference Optimization                                                                       |
| **Use Case**   | Guide the model toward generating more human-like responses                                          |

## Why fine-tuning for Human-Like-DPO-Dataset?

By fine-tuning a model specifically for Human-Like Response, we can:

- Improve conversational coherence.
- Reduce mechanical or impersonal responses.
- Enhance emotional intelligence in dialogue systems.

## Lab Structure

| Section                                 | Description                                                 | Duration   |
| --------------------------------------- | ----------------------------------------------------------- | ---------- |
| [Data Preparation](01-data-preparation) | Prepare and upload dataset to S3, create SageMaker Datasets | ~10 min    |
| [Fine-Tuning](02-fine-tuning)           | Launch serverless DPO job with LoRA                         | ~30-45 min |
| [Evaluation](03-evaluation)             | Evaluate with LLM-as-a-Judge and custom metrics             | ~45-60 min |
| [Deployment](04-deployment)             | Deploy to SageMaker real-time endpoint                      | ~10 min    |

:::alert{header="Important" type="warning"}
This workshop is designed to demonstrate serverless model customization, not to produce a production-grade model. We use ~3000 examples from the original dataset for fine-tuning. You can adapt this codebase to fine-tune the model using a larger dataset as needed.
:::
