---
title: "3. Prerequisites"
weight: 4
---

In this section, we are setting up the development environment we are going to use for the labs.

## Setup your development environment

In the AWS Console, search for **SageMaker AI**:

:image[]{src="/static/images/general/43-aws-console.png" height=384}

In the left menu, click on **Domains**:

:image[]{src="/static/images/general/44-studio-console.png" height=384}

Select the Studio domain:

:image[]{src="/static/images/general/45-domain-selection.png" height=384}

Click on **User profiles**:

:image[]{src="/static/images/general/46-user-tab.png" height=384}

In the dropdown menu of the user profile, click **Studio**:

:image[]{src="/static/images/general/47-open-studio.png" height=384}

Click on the **JupyterLab** icon on the top left:

:image[]{src="/static/images/general/40-jl-icon.png" height=384}

Click on **Open**:

:image[]{src="/static/images/general/10-studio-open-jupyterlab.png" height=384}

## Clone the lab repository

Once you're inside JupyterLab, open a new terminal:

:image[]{src="/static/images/general/open-terminal.png" height=384}

Clone the GitHub repository that contains all lab notebooks, training scripts, and configuration files:

```bash
git clone https://github.com/aws-samples/generative-ai-on-amazon-sagemaker.git
```

The workshop content is located under:

```
generative-ai-on-amazon-sagemaker/workshops/serverless-model-customization-with-sagemaker-ai/
```
