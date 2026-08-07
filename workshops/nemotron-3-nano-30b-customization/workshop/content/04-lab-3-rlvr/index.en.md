---
title: "Lab 3️⃣: Reinforcement Learning from Verifiable Rewards (RLVR)"
weight: 5
---

In this lab, you will fine-tune a model using **Reinforcement Learning from Verifiable Rewards (RLVR)** on Amazon SageMaker AI.

RLVR uses a reward function to score model-generated responses and a reinforcement learning algorithm (GRPO) to optimize the model. It is ideal for tasks with objectively correct or verifiable answers — instead of providing ideal completions (like SFT), you provide only the question and the correct answer, and let the model discover its own reasoning path through trial and reward.

:image[Reinforcement Learning concept]{src="/static/images/lab-3-rlvr/rl_concept.jpeg" height=384}

## Choose your path

This lab offers **two options**. Both fine-tune on the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) math dataset and follow the same data-prep → train → evaluate → deploy workflow. They differ only in how the reward signal is computed.

| Option                                                             | Reward signal                                               | Base model  | Best if you want to…                                         |
| ------------------------------------------------------------------ | ----------------------------------------------------------- | ----------- | ------------------------------------------------------------ |
| [**Option A — Built-in Reward**](01-option-a-builtin-reward)       | Built-in `default_compute_score` (exact match)              | Qwen 3 0.6B | Learn the core RLVR workflow with zero reward-function code  |
| [**Option B — Custom Reward Function**](02-option-b-custom-reward) | Your own Python reward function, registered as an Evaluator | Qwen 3 0.6B | Design and register a custom reward with shaping and metrics |

:::alert{header="Which should I do?" type="info"}
**Start with Option A** if you are new to RLVR — it covers the end-to-end workflow with the built-in reward and no extra code. **Move to Option B** when you want to control the scoring logic: it adds one step (authoring and registering a Python reward function) and passes it to the trainer via `custom_reward_function`. The options are independent — you can run either one on its own.
:::

:::alert{header="Important" type="warning"}
This workshop is designed to demonstrate serverless model customization, not to produce a production-grade model. We use a small subset of the GSM8K dataset for training. You can adapt this codebase to train with a larger dataset as needed.
:::
