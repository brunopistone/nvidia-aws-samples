---
title: "Lab 2️⃣: Inference"
weight: 30
---

In this lab, you will deploy the model you fine-tuned in [Lab 1](/02-lab-sft/) and run inference against it. You will explore two deployment paths: a **SageMaker AI real-time endpoint** and **Amazon Bedrock Custom Model Import**.

Once a model is customized, deployment is what turns it into a usable service. You will register the fine-tuned model, provision the serving stack, and test it end-to-end.

## What you will learn

1. **Deploy to SageMaker AI** - Register the merged fine-tuned model and create a real-time endpoint
2. **Run inference** - Invoke the endpoint with streaming and read the chain-of-thought separately from the answer
3. **Deploy to Amazon Bedrock** - Import the customized model for fully serverless inference
4. **Clean up resources** - Tear down endpoints to avoid ongoing charges

## Model and Serving

| Component         | Details                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------- |
| **Base Model**    | NVIDIA Nemotron 3 Nano 30B-A3B (`huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16`)  |
| **Customization** | Supervised Fine-Tuning with LoRA (from [Lab 1](/02-lab-sft/))                                 |
| **Serving Stack** | SageMaker **vLLM** Deep Learning Container with a custom `nano_v3` reasoning parser            |
| **Instance**      | `ml.g5.12xlarge` (4x NVIDIA A10G), tensor parallel size 4                                     |
| **Use Case**      | Multilingual chain-of-thought: reason in a target non-English language, answer in English      |

## Why deployment matters

Fine-tuning produces a set of model weights, but those weights only deliver value once they are served behind an endpoint. In this lab you will:

- Provision managed infrastructure for real-time inference
- Configure the serving container so the reasoning is parsed out of the raw output
- Validate that the deployed model reproduces the `<think>...</think>` behaviour it learned during training

## Lab Structure

| Section                             | Description                                                          | Duration   |
| ----------------------------------- | -------------------------------------------------------------------- | ---------- |
| [SageMaker Inference](01-sagemaker) | Deploy to a SageMaker real-time endpoint and run streaming inference  | ~20-25 min |
| [Bedrock Deployment](02-bedrock)    | Import the customized model into Amazon Bedrock                       | ~15-20 min |

:::alert{header="Important" type="warning"}
Remember to clean up your resources after completing the workshop to avoid ongoing charges. Real-time endpoints incur costs for as long as they are running. See the [Clean Up](/04-cleanup/) section for instructions.
:::
