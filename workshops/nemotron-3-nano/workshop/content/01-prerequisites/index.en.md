---
title: "Prerequisites"
weight: 10
---

## Prepare your workshop environment

Before starting the labs, you need access to an AWS account, a SageMaker AI Studio user profile and execution role, and a JupyterLab space with the workshop notebooks. Familiarity with Python, Jupyter notebooks, S3 paths, IAM roles, and supervised fine-tuning will help you follow the exercises. You do not need to provision a GPU cluster manually.

## Setup sequence

1. At an AWS-led event, follow [Workshop Studio event access](/01-prerequisites/1-workshop/) to sign in to the supplied account.
2. If you are working independently, follow [Your own AWS account](/01-prerequisites/2-account/) to prepare the account, Region, permissions, model access, and service quotas.
3. Follow [SageMaker AI Studio](/01-prerequisites/3-sagemaker/) to open JupyterLab, locate the automatically cloned workshop directory, select the Python 3 kernel, and install notebook dependencies. Start in `02-smtj-workshop/` for the primary 4B journey.

The primary journey is [SageMaker Training jobs: 4B](/03-training-jobs/), which is standalone after this setup. [Optional: Serverless model customization](/02-serverless/) remains available as an alternative extension, not a prerequisite. Review the [workflow comparison](/) before attempting both, since their default S3 keys overlap.
