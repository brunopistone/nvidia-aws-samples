---
title: "Fine-tune NVIDIA Nemotron 3 model on Amazon SageMaker AI serverless Model Customization"
weight: 0
---

Welcome to the **Fine-tune NVIDIA Nemotron 3 model on Amazon SageMaker AI serverless Model Customization** workshop!

In this hands-on workshop, you'll experience how Amazon SageMaker AI serverless model customization changes the way developers fine-tune foundation models. Starting from the pre-trained **NVIDIA Nemotron 3 Nano 30B-A3B** model, you will prepare a dataset, run a Supervised Fine-Tuning (SFT) job with LoRA, evaluate the improvements, and deploy the customized model for inference — all without provisioning or managing any training infrastructure.

Working directly in SageMaker AI Studio JupyterLab, you'll take an open-weight NVIDIA Nemotron 3 model through the complete customization lifecycle: data preparation, serverless fine-tuning, evaluation, and deployment.

## The task

You will fine-tune the model on **automated contract review** using the [ContractNLI](https://stanfordnlp.github.io/contract-nli/) dataset: given a non-disclosure agreement and a fixed checklist of 17 legal hypotheses, the model must classify each hypothesis as `Entailment`, `Contradiction`, or `NotMentioned`, and cite the specific contract clauses (spans) that justify each decision.

This is a structured reasoning task — the output is strict JSON with one entry per hypothesis. The model must read the full contract and reason per-item, not retrieve a single passage per question.

## What You'll Learn

- How to use SageMaker AI serverless Model Customization to fine-tune foundation models
- How to customize NVIDIA Nemotron 3 models without any infrastructure provisioning or management
- How to apply Supervised Fine-Tuning (SFT) with LoRA to a selected NVIDIA Nemotron 3 model
- How to evaluate fine-tuned models with automated LLM-as-a-Judge metrics tailored to your use case
- How to serve the customized model on a SageMaker real-time endpoint with vLLM

::alert[**Important note** This workshop uses the **SFT** fine-tuning technique and walks through the full model customization lifecycle: data preparation, training, evaluation, and deployment. Each lab provides guided instructions so you can follow along step-by-step. The [Prerequisites](/00-prerequisites/) module is mandatory before running any other module.]{type="info"}

### Workshop modules

| Module | ⏰ Duration | Level | Target Audience |
|--------|----------|-------|-----------------|
| 1. [Prerequisites](/00-prerequisites/) | 10-15 mins | Basic | All participants preparing setup and environment access |
| 2. [Lab: Supervised Fine-Tuning (SFT)](/01-data-preparation/) | 50-60 mins | Advanced | Data scientists and AI practitioners |
| 3. [Lab: Inference](/04-lab-inference/) | 40-50 mins | Advanced | ML engineers and data scientists |
| 4. [Clean Up](/05-cleanup/) | 5-10 mins | Basic | All participants |
| 5. [Summary](/06-summary/) | 5 mins | Basic | All participants |

### How to run the workshop

This workshop follows a hands-on, self-paced format. Each module walks through Jupyter notebooks that you run in your own JupyterLab environment (setup instructions are in the prerequisites section). The notebooks include:

- Step-by-step instructions and explanations
- Code samples that you can run and modify
- Links to additional resources

### Workshop GitHub repository

The workshop notebooks are available in the public [nvidia-aws-samples](https://github.com/NVIDIA/nvidia-aws-samples) GitHub repository, under `workshops/nemotron-3-nano-30b-customization/code/`.

We welcome you to bookmark and star the repository for access to future content we publish.

### Disclaimers

::alert[All code is covered under the [MIT-0 license](https://github.com/aws/mit-0)]
::alert[Please remember to clean up all resources created during this workshop to avoid ongoing charges to your AWS account.]

### Security Best Practices

Throughout this workshop, we adhere to AWS service security best practices. We encourage you to familiarize yourself with the [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/) and apply them in your own implementations. Key points include:

- Using IAM roles and policies with least privilege
- Encrypting data at rest and in transit
- Implementing network security controls
- Regularly monitoring and auditing your resources

Remember to always follow security best practices when working with AWS services and sensitive data.
