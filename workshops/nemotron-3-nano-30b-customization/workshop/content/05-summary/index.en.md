---
title: "Summary"
weight: 50
---

In this workshop, you fine-tuned **NVIDIA Nemotron 3 Nano 30B-A3B** using Amazon SageMaker AI serverless model customization on the [ContractNLI](https://stanfordnlp.github.io/contract-nli/) contract review dataset. Starting from the pre-trained Nemotron Nano foundation model, you:

- Prepared the ContractNLI dataset and registered it as SageMaker AI Datasets
- Ran a Supervised Fine-Tuning (SFT) job with LoRA — without provisioning or managing any training infrastructure
- Taught the model to classify each of 17 legal hypotheses as Entailment, Contradiction, or NotMentioned, and cite the supporting contract clauses
- Evaluated the customized model with a custom statistical scorer and LLM-as-a-Judge, reaching parity with Claude Sonnet 5 on this task
- Deployed it for real-time inference on a SageMaker endpoint with the vLLM container and a custom reasoning parser

## Key takeaways

- **Serverless customization removes the infrastructure decision.** You pick a base model, a technique and a dataset; SageMaker AI provisions the compute.
- **Datasets and models are first-class registry assets.** Registering them once makes them reusable and traceable across training, evaluation and deployment jobs.
- **Generic metrics aren't enough.** The custom scorer measured what the task actually required — contradiction detection, evidence citation, and structured JSON output.
- **Serving configuration matters.** The `nano_v3` reasoning parser is what separates the chain-of-thought from the final answer at inference time.

## For more on serverless model customization in Amazon SageMaker AI

Your learning does not have to stop here. Bookmark these resources and use them as references as you apply these techniques to your own use cases.

- AWS product page: [SageMaker AI model customization](https://aws.amazon.com/sagemaker/ai/model-customization/)
- AWS documentation: [Customize a model in Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customization.html)
- Related AWS workshop: [Serverless Model Customization with Amazon SageMaker AI](https://catalog.us-east-1.prod.workshops.aws/workshops/548b5be9-2da8-4c93-82f7-b0b474108ab3/en-US)
- NVIDIA: [Nemotron open models](https://developer.nvidia.com/nemotron)
- Code: [NVIDIA/nvidia-aws-samples](https://github.com/NVIDIA/nvidia-aws-samples) on GitHub

## Call to Action

- Apply the concepts from this workshop to your own use cases on AWS.
- Explore [Amazon SageMaker AI](https://aws.amazon.com/sagemaker/ai/) and dive deeper in the documentation.
- Follow [What's new at AWS](https://aws.amazon.com/new/) for the latest launches.

::alert[**Did you clean up?** Make sure you completed the [Clean Up](/04-cleanup/) module — a running endpoint keeps billing.]{type="warning"}

::alert[**Congratulations** on completing the workshop. We appreciate and value your time and interest.]{type="success"}

::alert[**IMPORTANT** - Please remember to fill out the survey!]{type="info"}

**Thank you!**
