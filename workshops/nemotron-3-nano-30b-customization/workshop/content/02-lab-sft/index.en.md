---
title: "Lab 1: Supervised Fine-Tuning (SFT)"
weight: 20
---

In this lab, you will fine-tune **NVIDIA Nemotron 3 Nano 30B-A3B** using Serverless Supervised Fine-Tuning on Amazon SageMaker AI.

SFT is the most straightforward customization technique: you provide prompt/completion pairs and the model learns to produce the desired outputs. It's ideal for teaching a model a new output format, a new style, or a structured reasoning behaviour.

:image[Supervised Fine-Tuning concept]{src="/static/images/lab-1-sft/sft_concept.jpeg" height=384}

## What you will learn

1. **Prepare your dataset** - Format data for SFT and create SageMaker AI Datasets
2. **Run a serverless fine-tuning job** - Use the `SFTTrainer` API with LoRA
3. **Evaluate your model** - Use LLM-as-a-Judge with task-specific custom metrics
4. **Deploy your model** - Serve the customized model (covered in [Lab 2](/03-lab-inference/))

## Model and Dataset

| Component | Details |
| -------------- | ------------------------------------------------------------------------------------------------------------- |
| **Base Model** | NVIDIA Nemotron 3 Nano 30B-A3B (`huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16`) |
| **Dataset** | [ContractNLI](https://stanfordnlp.github.io/contract-nli/): 607 real NDAs annotated against 17 legal hypotheses, CC-BY-4.0 |
| **Technique** | Supervised Fine-Tuning with LoRA |
| **Use Case** | Contract review: classify each hypothesis as Entailment / Contradiction / NotMentioned and cite evidence spans |

## The task: automated contract review

The model reads a non-disclosure agreement and checks it against a fixed 17-item legal checklist. For each hypothesis on the checklist, it must:

1. **Classify** the hypothesis as `Entailment`, `Contradiction`, or `NotMentioned`
2. **Cite evidence**: the specific numbered clause (span) in the contract that justifies each Entailment or Contradiction verdict

The output is strict JSON: one entry per hypothesis, always in the same format.

This makes a precise fine-tuning target because:

- **It is fully verifiable**: you can score it programmatically against expert annotations
- **It requires reading, not retrieval**: any of the 17 items could be decided by any clause
- **The base model does it poorly**: detecting contradictions and citing specific clauses is where the improvement is visible and measurable

## Why fine-tune for this?

Prompting alone does not solve it. Adding worked examples to the prompt makes the small base model *worse* at detecting contradictions. Demonstrations teach output shape, but shape is not the bottleneck. The model's problem is reading the contract carefully enough to distinguish a clause that conflicts with a hypothesis from one that simply doesn't address it. That requires task-specific weight updates, not more examples.

## Lab Structure

| Section | Description | Duration |
| --------------------------------------- | ----------------------------------------------------------- | ---------- |
| [Data Preparation](01-data-preparation) | Prepare and upload the dataset to S3, create SageMaker AI Datasets | ~10 min |
| [Fine-Tuning](02-fine-tuning) | Launch the serverless SFT job with LoRA | ~20-30 min |
| [Evaluation](03-evaluation) | Evaluate with LLM-as-a-Judge and custom metrics | ~20-25 min |

:::alert{header="Important" type="warning"}
This workshop demonstrates serverless model customization on a real, annotated dataset. ContractNLI contains 607 NDAs split into 423 train / 61 dev / 123 test documents, small enough that the fine-tuning job runs in under 30 minutes, but large enough to produce a measurable improvement. You can adapt this codebase to a larger contract dataset by changing the data preparation notebook.
:::
