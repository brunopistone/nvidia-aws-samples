---
title: "Sagemaker AI Studio"
weight: 13
---

## Start SageMaker AI Studio

After signing into the AWS account, follow these instructions to open the SageMaker Studio environment.

1. In the AWS console, navigate to the **Amazon SageMaker AI** console — start typing `SageMaker AI` in the search box at the top.

![01-aws-console-sagemaker](/static/images_prereq/01-aws-console-sagemaker.png)

2. In the left menu, under **Applications and IDEs**, select **Studio**.
3. Select your user profile and choose **Open Studio**. The Studio UI opens in a new browser tab.

![01-aws-sagemaker-studiouser](/static/images_prereq/01-aws-sagemakerstudio-user-console.png)

## Open a JupyterLab space

At an AWS-led event, a JupyterLab space is already created for you. Otherwise, create a new JupyterLab space with the defaults, following [the Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide.html).

1. Select the **JupyterLab** app in the top left.

![01-studio-jupyterlab-app](/static/images_prereq/01-studio-jupyterlab-app.png)

2. If the space is stopped, choose **Run** to start it.

![sagemaker_studio_space_run](/static/images_prereq/sagemaker_studio_space_run.png)

3. Once the space is running, choose **Open** to launch the JupyterLab application.

![sagemaker_studio_space_open](/static/images_prereq/sagemaker_studio_space_open.png)

## Find the workshop notebooks

The event setup clones the [NVIDIA/nvidia-aws-samples](https://github.com/NVIDIA/nvidia-aws-samples) repository into your space. The notebooks for this workshop are in:

```
nvidia-aws-samples/workshops/nemotron-3-nano-30b-customization/code/
```

If the folder is not there, open a terminal in JupyterLab and clone it yourself:

```bash
git clone https://github.com/NVIDIA/nvidia-aws-samples.git
cd nvidia-aws-samples/workshops/nemotron-3-nano-30b-customization/code
```

You will run these notebooks in order:

| Notebook | Module |
| ----------------------------- | ------------------------------------------------------ |
| `1-prepare-data.ipynb` | [Data Preparation](/02-lab-sft/01-data-preparation/) |
| `2-fine-tune-llm.ipynb` | [Fine-Tuning](/02-lab-sft/02-fine-tuning/) |
| `3-evaluation.ipynb` | [Evaluation](/02-lab-sft/03-evaluation/) |
| `4-deployment.ipynb` | [SageMaker Inference](/03-lab-inference/01-sagemaker/) |

Select the **`Python 3 (ipykernel)`** kernel when a notebook asks, and run the first cell to install the dependencies from `requirements.txt`.

::alert[**Ready!** SageMaker Studio is now configured and you are all set to dive into the labs, starting with [Supervised Fine-Tuning](/02-lab-sft/).]{type="success"}
