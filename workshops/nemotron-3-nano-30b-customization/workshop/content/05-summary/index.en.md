---
title: "📝 Summary"
weight: 50
---

In this workshop, you fine-tuned **NVIDIA Nemotron 3 Nano 30B-A3B** using Amazon SageMaker AI serverless model customization. Starting from the pre-trained Nemotron Nano foundation model, you:

- Prepared the [Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) dataset and registered it as SageMaker AI Datasets
- Ran a Supervised Fine-Tuning (SFT) job with LoRA — without provisioning or managing any training infrastructure
- Taught the model to reason inside `<think>...</think>` tags in a target non-English language and answer in English
- Evaluated the customized model with LLM-as-a-Judge, combining built-in metrics with four task-specific custom metrics
- Deployed it for real-time inference on a SageMaker endpoint with the vLLM container and a custom reasoning parser

## Key takeaways

- **Serverless customization removes the infrastructure decision.** You pick a base model, a technique and a dataset; SageMaker AI provisions the compute.
- **Datasets and models are first-class registry assets.** Registering them once makes them reusable and traceable across training, evaluation and deployment jobs.
- **Generic metrics aren't enough.** The custom metrics you defined measured the behaviour you actually fine-tuned for — reasoning language, answer language, and output structure.
- **Serving configuration matters.** The `nano_v3` reasoning parser is what separates the chain-of-thought from the final answer at inference time.

## For more on serverless model customization in Amazon SageMaker AI

Your learning does not have to stop here. Bookmark these resources and use them as references as you apply these techniques to your own use cases.

- 📚 AWS product page: [SageMaker AI model customization](https://aws.amazon.com/sagemaker/ai/model-customization/)
- 📖 AWS documentation: [Customize a model in Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customization.html)
- ☁️ Related AWS workshop: [Serverless Model Customization with Amazon SageMaker AI](https://catalog.us-east-1.prod.workshops.aws/workshops/548b5be9-2da8-4c93-82f7-b0b474108ab3/en-US)
- 🧠 NVIDIA: [Nemotron open models](https://developer.nvidia.com/nemotron)
- 💻 Code: [NVIDIA/nvidia-aws-samples](https://github.com/NVIDIA/nvidia-aws-samples) on GitHub

## 📌 Call to Action

- Apply the concepts from this workshop to your own use cases on AWS.
- Explore [Amazon SageMaker AI](https://aws.amazon.com/sagemaker/ai/) and dive deeper in the documentation.
- Follow [What's new at AWS](https://aws.amazon.com/new/) for the latest launches.

::alert[**Did you clean up?** Make sure you completed the [Clean Up](/04-cleanup/) module — a running endpoint keeps billing.]{type="warning"}

::alert[**Congratulations** on completing the workshop. We appreciate and value your time and interest.]{type="success"}

::alert[**IMPORTANT** - Please remember to fill out the survey! Your 5 Stars keep us motivated 🥳]{type="info"}

**Thank you!**
