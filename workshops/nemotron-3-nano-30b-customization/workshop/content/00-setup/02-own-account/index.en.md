---
title: "2. Getting started in your own AWS account"
weight: 3
---

If you would like to run this workshop in your own AWS account, follow the instructions below.

## Setup Instructions

The workshop infrastructure can be deployed using the following CloudFormation template:

:button[Deploy SageMaker Studio domain]{variant="primary" href="https://console.aws.amazon.com/cloudformation/home?#/stacks/quickcreate?templateURL=https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/548b5be9-2da8-4c93-82f7-b0b474108ab3/cfn/workshop-setup.yaml&stackName=serverless-model-customization-stack" external="true"}

The CloudFormation template requires the following parameters:

1. Private Networking environment: VPC, Subnets, Security groups
2. IAM Roles: IAM Policy and role for Amazon SageMaker Studio
3. Amazon SageMaker Studio domain
4. Amazon SageMaker Studio user profile
5. Managed MLflow tracking server

The CloudFormation template by default will deploy resources in `us-west-2`. If you want to operate in a different region, make sure to update the **Parameters** related to the Availability Zones.

## Estimated Cost

Prices refer to the **US East (N. Virginia)** region. For the latest pricing, refer to the [Amazon SageMaker AI pricing page](https://aws.amazon.com/sagemaker-ai/pricing/).

### Training (On-Demand)

| Lab              | Technique    | Model                      |   Price | Unit          |
| ---------------- | ------------ | -------------------------- | ------: | ------------- |
| Lab 1            | SFT / LoRA   | Qwen3 4B                   |  $0.500 | per 1M tokens |
| Lab 2            | DPO / LoRA   | Meta Llama 3.2 1B Instruct |  $0.416 | per 1M tokens |
| Lab 3 — Option A | RLVR / LoRA  | Qwen3 0.6B                 | $80.000 | per hour      |
| Lab 3 — Option B | RLVR / LoRA  | Qwen3 0.6B                 | $80.000 | per hour      |
| Lab 4            | RLAIF / LoRA | Qwen2.5 7B Instruct        | $80.000 | per hour      |

### Evaluation (LLM-as-a-Judge)

| Model                      | Input Token Price | Output Token Price | Unit          |
| -------------------------- | ----------------: | -----------------: | ------------- |
| Qwen2.5 7B Instruct        |            $0.200 |             $0.200 | per 1M tokens |
| Meta Llama 3.2 1B Instruct |            $0.100 |             $0.100 | per 1M tokens |

### Additional Costs

Other AWS services used during the workshop may incur costs, including:

- **Amazon SageMaker Studio** (JupyterLab space)
- **Amazon SageMaker Endpoints** (model deployment and inference)
- **Amazon S3** (dataset and artifact storage)
- **MLflow Tracking Server** (experiment tracking)
- **Amazon Bedrock** (reward model invocations in Lab 4)

:::alert{header="Important" type="warning"}
Remember to clean up all resources after completing the workshop to avoid ongoing charges. Each lab includes a cleanup section with instructions.
:::
