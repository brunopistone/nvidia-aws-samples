---
title: "Serverless Model Customization with Amazon SageMaker AI"
weight: 0
---

:image[SageMaker AI]{src="/static/images/sagemaker.jpeg" height=384}

---

Welcome to the _Serverless Model Customization with Amazon SageMaker AI_ workshop.

In this workshop, you will learn how to:

1. Use the new serverless customization capability in Amazon SageMaker AI
2. Fine-tune popular AI models with just a few clicks—no infrastructure management required
3. Deploy customized models to Amazon Bedrock or SageMaker endpoints
4. Evaluate your customized models using built-in tools

# What is Serverless Model Customization?

Serverless customization in Amazon SageMaker AI provides an easy-to-use interface for the latest fine-tuning techniques. You can accelerate the AI model customization process from months to days—all entirely serverless so you can focus on model tuning rather than managing infrastructure.

When you choose serverless customization, SageMaker AI automatically selects and provisions the appropriate compute resources based on the model and data size. You only pay for the tokens processed during training and inference.

## Supported Models

Serverless customization supports popular AI models including:

- **Google Gemma 4** - Lightweight open models from Google
- **Qwen** - Multilingual models from Alibaba
- **NVIDIA Nemotron 3** - Open reasoning models from NVIDIA
- **Amazon Nova** - Amazon's foundation models
- **Meta Llama** - Open-source LLMs from Meta
- **DeepSeek** - Advanced reasoning models
- **GPT-OSS** - Open-source GPT variants

## Customization Techniques

SageMaker AI supports multiple customization techniques, each suited for different stages of the model training process:

:image[Model customization techniques]{src="/static/images/general/customization_techniques.jpg" height=384}

| Technique                                                 | Description                                                 |
| --------------------------------------------------------- | ----------------------------------------------------------- |
| **Supervised Fine-Tuning (SFT)**                          | Train models on labeled input-output pairs                  |
| **Direct Preference Optimization (DPO)**                  | Align models with human preferences without reward modeling |
| **Reinforcement Learning from Verifiable Rewards (RLVR)** | Optimize models using verifiable reward signals             |
| **Reinforcement Learning from AI Feedback (RLAIF)**       | Use AI-generated feedback for model alignment               |

All customization techniques in this workshop use **Parameter Efficient Fine-Tuning (PEFT)** with **LoRA** adapters. Instead of updating all model parameters, LoRA trains a small set of low-rank weight matrices — dramatically reducing GPU memory requirements and training cost while maintaining model quality.

:image[Full Fine-Tuning vs PEFT/LoRA]{src="/static/images/general/peft_lora_concept.jpeg" height=384}

# Objective of this Workshop

Learn how to customize AI models using the new serverless capability in Amazon SageMaker AI. Through hands-on exercises, you'll:

- Select and configure models for customization
- Prepare and upload training datasets
- Launch serverless fine-tuning jobs
- Monitor training progress with MLflow
- Deploy customized models for inference
- Evaluate model performance

# Workshop Structure

1. **Setup** - Access SageMaker Studio and explore the Models interface
2. **Prepare Dataset** - Format your training data for the selected customization technique
3. **Customize Model** - Launch a serverless fine-tuning job using the UI
4. **Deploy Model** - Deploy your customized model to Amazon Bedrock or SageMaker
5. **Evaluate** - Test and compare your customized model against the base model

# Key Benefits

- **No infrastructure management** - Focus on model tuning, not compute provisioning
- **Pay-per-token pricing** - Only pay for tokens processed during training
- **Built-in MLflow integration** - Automatic experiment tracking and visualization
- **Flexible deployment** - Deploy to Bedrock for serverless inference or SageMaker for custom endpoints
- **Multiple techniques** - Choose from SFT, DPO, RLVR, or RLAIF based on your use case

# Prerequisites

To get the most out of this workshop, you should have:

- **AWS Familiarity** - Basic understanding of AWS services and console navigation
- **Machine Learning Basics** - Understanding of fundamental ML concepts like training and evaluation
- **Data Preparation** - Familiarity with preparing datasets in JSON/JSONL format
