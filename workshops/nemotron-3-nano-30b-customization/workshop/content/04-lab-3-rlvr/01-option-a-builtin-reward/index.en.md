---
title: "Option A — Built-in Reward"
weight: 1
---

In this option, you will learn how to fine-tune **Qwen 3 0.6B** using Reinforcement Learning from Verifiable Rewards (RLVR) on Amazon SageMaker AI, using SageMaker's **built-in reward function**.

RLVR uses a reward function to score model-generated responses and reinforcement learning algorithms (like GRPO) to optimize the model. SageMaker AI provides built-in reward functions — exact match, code execution, and math — out of the box, and also supports custom reward functions for more advanced verification logic (see [Option B](../02-option-b-custom-reward)). This makes RLVR ideal for tasks with objectively correct or verifiable answers.

:image[Reinforcement Learning concept]{src="/static/images/lab-3-rlvr/rl_concept.jpeg" height=384}

## What you will learn

1. **Prepare your dataset** - Format data for RLVR with verifiable answers and create SageMaker AI Datasets
2. **Run a serverless RLVR training job** - Use the RLVRTrainer API
3. **Evaluate your model** - Use the MATH benchmark to measure reasoning improvements
4. **Deploy your model** - Create a SageMaker real-time endpoint

## Model and Dataset

| Component      | Details                                                               |
| -------------- | --------------------------------------------------------------------- |
| **Base Model** | Qwen 3 0.6B (`huggingface-reasoning-qwen3-06b`)                       |
| **Dataset**    | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (180 samples)   |
| **Technique**  | Reinforcement Learning from Verifiable Rewards (RLVR)                 |
| **Use Case**   | Mathematical reasoning                                                |

## Why RLVR for mathematical reasoning?

RLVR is ideal for tasks with objectively verifiable answers. Instead of providing ideal completions (like SFT), RLVR:

- Provides only the question and the correct answer
- Lets the model discover its own reasoning path through trial and reward
- Uses a rule-based reward function to verify correctness automatically
- Requires no human feedback or separate reward model

## Lab Structure

| Section                                 | Description                                                       | Duration   |
| --------------------------------------- | ----------------------------------------------------------------- | ---------- |
| [Data Preparation](01-data-preparation) | Prepare and upload GSM8K dataset to S3, create SageMaker Datasets | ~10 min    |
| [Fine-Tuning](02-fine-tuning)           | Launch serverless RLVR training job                               | ~15-20 min |
| [Evaluation](03-evaluation)             | Evaluate with MATH benchmark (base vs fine-tuned)                 | ~15-20 min |
| [Deployment](04-deployment)             | Deploy to SageMaker real-time endpoint                            | ~10-15 min |

:::alert{header="Important" type="warning"}
This workshop is designed to demonstrate serverless model customization, not to produce a production-grade model. We use a small subset (180 examples) of the GSM8K dataset for training. You can adapt this codebase to train with a larger dataset as needed.
:::
