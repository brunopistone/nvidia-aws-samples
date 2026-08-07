---
title: "Option B — Custom Reward Function"
weight: 2
---

This option extends [Option A](../01-option-a-builtin-reward) by replacing the built-in rule verifier with a **custom Python reward function** that you author and register as a SageMaker AI Registry Evaluator.

You will fine-tune **Qwen 3 0.6B** using Reinforcement Learning from Verifiable Rewards (RLVR) on the GSM8K math dataset — the same technique and dataset as Option A — but this time the reward signal comes from your own scoring logic.

:image[Reinforcement Learning concept]{src="/static/images/lab-3-rlvr/rl_concept.jpeg" height=384}

:::alert{header="New to RLVR?" type="info"}
Start with [Option A](../01-option-a-builtin-reward) — it covers the core RLVR workflow with the built-in reward. This option focuses on the one thing that changes: bringing your own reward function. You can also run it independently if your goal is custom reward design.
:::

## Why a custom reward function?

The built-in `default_compute_score` (exact match) is enough for tasks like GSM8K where the answer is a single number. But many real tasks need richer scoring — partial credit, format checks, reasoning quality, or domain-specific rules. A custom reward function lets you:

- **Score on multiple signals** — combine numeric correctness with format and reasoning metrics
- **Shape the reward** — avoid all-zero rewards on early rollouts so training has a gradient to follow
- **Emit observability metrics** — surface per-record metrics (correctness, parseability, format) in MLflow
- **Encode domain rules** — apply any Python logic the task requires

In this option, the reward favors numerically correct final answers while adding a small shaping signal for answer parseability, `####` format, and the presence of step-by-step reasoning.

## What you will learn

1. **Prepare your dataset** - Format GSM8K for a custom-reward RLVR job with stable IDs and reference answers
2. **Create the reward function** - Author, smoke-test, and register a Python reward function as a `REWARD_FUNCTION` Evaluator
3. **Run a serverless RLVR training job** - Pass your evaluator to the `RLVRTrainer` via `custom_reward_function`
4. **Evaluate your model** - Use the MATH benchmark to measure reasoning improvements
5. **Deploy your model** - Create a SageMaker real-time endpoint

## Model and Dataset

| Component      | Details                                                                       |
| -------------- | ----------------------------------------------------------------------------- |
| **Base Model** | Qwen 3 0.6B (`huggingface-reasoning-qwen3-06b`)                               |
| **Dataset**    | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (180 samples)           |
| **Technique**  | Reinforcement Learning from Verifiable Rewards (RLVR)                         |
| **Reward**     | Custom Python reward function registered as a SageMaker AI Registry Evaluator |
| **Use Case**   | Custom reward design for mathematical reasoning                               |

## How it differs from Option A

| Aspect            | Option A (Built-in Reward)       | Option B (Custom Reward Function)                      |
| ----------------- | -------------------------------- | ------------------------------------------------------ |
| Reward            | Built-in `default_compute_score` | Custom `reward_function.py` registered as an Evaluator |
| Extra step        | —                                | "Create the Reward Function" section                   |
| Trainer parameter | —                                | `custom_reward_function` passed to `RLVRTrainer`       |

## Lab Structure

| Section                                          | Description                                                          | Duration   |
| ------------------------------------------------ | -------------------------------------------------------------------- | ---------- |
| [Data Preparation](01-data-preparation)          | Prepare GSM8K with stable IDs and reference answers, create Datasets | ~10 min    |
| [Create the Reward Function](02-reward-function) | Author, smoke-test, and register a custom Python reward function     | ~10 min    |
| [Fine-Tuning](03-fine-tuning)                    | Launch a serverless RLVR job with `custom_reward_function`           | ~15-20 min |
| [Evaluation](04-evaluation)                      | Evaluate the fine-tuned model with the MATH benchmark                | ~15-20 min |
| [Deployment](05-deployment)                      | Deploy to SageMaker real-time endpoint                               | ~10-15 min |

:::alert{header="Important" type="warning"}
This workshop is designed to demonstrate serverless model customization, not to produce a production-grade model. We use a small subset of the GSM8K dataset for training. You can adapt this codebase to train with a larger dataset as needed.
:::
