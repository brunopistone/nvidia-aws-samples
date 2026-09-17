---
title: "Lab 2: Inference"
weight: 30
---

In this lab, you will deploy the model you fine-tuned in [Lab 1](/02-lab-sft/) to a **SageMaker AI real-time endpoint** and run inference against it.

Once a model is customized, deployment is what turns it into a usable service. You will register the fine-tuned model, provision the serving stack, and test it end-to-end.

## What you will learn

1. **Deploy to SageMaker AI** - Register the merged fine-tuned model and create a real-time endpoint
2. **Run inference** - Invoke the endpoint and verify the model still returns valid JSON for all 17 checklist items
3. **Clean up resources** - Tear down the endpoint to avoid ongoing charges

## Model and Serving

| Component | Details |
| ----------------- | -------------------------------------------------------------------------------------------- |
| **Base Model** | NVIDIA Nemotron 3 Nano 30B-A3B (`huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16`) |
| **Customization** | Supervised Fine-Tuning with LoRA (from [Lab 1](/02-lab-sft/)) |
| **Serving Stack** | AWS LMI/DJL container running vLLM via `djl_python.lmi_vllm.vllm_async_service` |
| **Instance** | `ml.g5.xlarge` (1x NVIDIA A10G) |
| **Use Case** | Contract review: serve the fine-tuned model and verify it still returns valid JSON for all 17 checklist items |

## Why deployment matters

Fine-tuning produces a set of model weights, but those weights only deliver value once they are served behind an endpoint. In this lab you will:

- Provision managed infrastructure for real-time inference
- Configure the serving container so the reasoning is parsed out of the raw output
- Validate that the deployed model reproduces the `<think>...</think>` behaviour it learned during training

## Lab Structure

| Section | Description | Duration |
| ----------------------------------- | -------------------------------------------------------------------- | ---------- |
| [SageMaker Inference](01-sagemaker) | Deploy to a SageMaker real-time endpoint and run inference | ~20-25 min |

:::alert{header="Important" type="warning"}
Remember to clean up your resources after completing the workshop to avoid ongoing charges. Real-time endpoints incur costs for as long as they are running. See the [Clean Up](/04-cleanup/) section for instructions.
:::
