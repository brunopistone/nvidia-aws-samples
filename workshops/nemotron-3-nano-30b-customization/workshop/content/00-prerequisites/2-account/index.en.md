---
title: "Self paced AWS account"
weight: 12
---

::alert[Do not use these instructions if you're participating in an AWS-led workshop event — a workshop AWS account is already provisioned with the required configuration. Skip this page and move to [SageMaker AI Studio](/00-prerequisites/3-sagemaker/).]

The instructions on this page are for running the serverless model customization workshop in **your own AWS account**.

::alert[Running this workshop in your own environment will incur costs. Please remember to clean up all resources created during the workshop to avoid ongoing charges to your AWS account.]{type="warning"}

## AWS Account access

You need an AWS account. If you don't already have one, follow the [Setting Up Your AWS Environment](https://aws.amazon.com/getting-started/guides/setup-environment/) getting started guide.

This workshop has been tested in the **`N. Virginia (us-east-1)`** region. Make sure the console is set to that region before you start.

## Required IAM permissions

The IAM execution role of the Studio user profile you use for the workshop needs the following managed policies:

```
AmazonSageMakerFullAccess
AmazonS3FullAccess
AmazonBedrockFullAccess
```

:::alert{header="Bedrock trust policy is mandatory" type="warning"}
The evaluation lab runs LLM-as-a-Judge scoring on Amazon Bedrock, which requires **`bedrock.amazonaws.com` to be a trusted entity in the execution role's trust policy** — not just the Bedrock managed policy.

Without it, the evaluation pipeline starts but the `EvaluateCustomModelMetrics` step fails. See [Amazon Bedrock permissions setup](https://docs.aws.amazon.com/bedrock/latest/userguide/judge-service-roles.html).
:::

## Model and service access

- **Amazon Bedrock model access** for the judge model `amazon.nova-pro-v1:0` (request it in the Bedrock console under **Model access**).
- **SageMaker JumpStart access** to the base model — the training job accepts the EULA programmatically with `accept_eula=True`.
- **Service quota for `ml.g5.12xlarge` for endpoint usage** (4x NVIDIA A10G), needed for the real-time deployment lab. Request an increase in **Service Quotas** → **Amazon SageMaker** if your account has none.
- Optional, only for the Bedrock deployment lab: quota increases for `Imported models per account` and `Concurrent model import jobs`.

## Amazon SageMaker AI Studio

To run the notebooks we strongly recommend [SageMaker AI Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated.html), which requires a [SageMaker AI domain](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-entity-status.html).

### Existing domain

If you already have a SageMaker AI domain, follow [Get your execution role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html#sagemaker-roles-get-execution-role) to find the IAM execution role used by your Studio user profile, then attach the policies listed above. You can also [create a new user profile](https://docs.aws.amazon.com/sagemaker/latest/dg/domain-user-profile-add.html) with a dedicated execution role just for this workshop.

### Provision a new domain

If you don't have a domain, or want a dedicated one for the workshop, create a new domain. You can have more than one domain in the same account and Region.

::alert[If you have more than one domain in your account, consider the limit on active domains per Region per account.]

**Option 1 — AWS Console:** follow [quick setup for Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html).

**Option 2 — CloudFormation:** this workshop ships the same template used for AWS-led events at `static/cfn/workshop-setup.yaml`. It creates a VPC, VPC endpoints, a KMS key, a Studio domain, a user profile with a correctly configured execution role, and a JupyterLab space that clones the workshop repository automatically.

:::alert{header="Required execution role policies" type="warning"}
If you create the domain manually via the AWS Console, remember to attach `AmazonSageMakerFullAccess`, `AmazonS3FullAccess` and `AmazonBedrockFullAccess` to the execution role of the user profile you use, **and** add `bedrock.amazonaws.com` to its trust policy.
:::

## Start SageMaker AI Studio

Once the account is set up, move on to [SageMaker AI Studio](/00-prerequisites/3-sagemaker/).
