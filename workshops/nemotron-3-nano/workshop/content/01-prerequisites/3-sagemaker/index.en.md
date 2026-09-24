---
title: "SageMaker AI Studio"
weight: 13
---

## 1. Open your Studio profile

Confirm the account and Region in the AWS console, then open **Amazon SageMaker AI**. Under **Applications and IDEs**, choose **Studio**, select the approved user profile, and choose **Open Studio**. Event participants should use the instructor-provided profile.

![Open SageMaker AI Studio](/static/images_prereq/01-aws-sagemakerstudio-user-console.png)

If the profile or domain is missing, ask the event instructor or complete [account setup](/01-prerequisites/2-account/) for your own environment.

## 2. Open the running JupyterLab space

For the AWS-hosted workshop, the JupyterLab app is provisioned and started automatically. From Studio **Home**, choose **JupyterLab** under **Applications**, or choose **View JupyterLab spaces** on the JupyterLab card. Find your instructor-provided private space. When its status is **Running**, choose **Open** to enter JupyterLab. You do not need to choose **Run** or create another space.

The [workshop CloudFormation template](/static/cfn/workshop-setup.yaml) creates the private space `<user-profile>-jupyterlab-space` and starts it with **SageMaker Distribution CPU 4.5.0**, `ml.m5.large`, and 20 GB of persistent EBS storage by default. An own-account environment created with this template also starts automatically. If the app is still starting, choose **Refresh** and wait for **Running**. If startup fails, ask your instructor or administrator to check the app status and provisioning logs.

### If your space is stopped

Open the existing space's settings, confirm **SageMaker Distribution CPU 4.5.0** and the workshop lifecycle configuration when using the template, then choose **Run**. Wait for **Running** and choose **Open**. If the spaces list is filtered to **Running**, clear that filter to find a stopped space. Use this startup step only for a stopped app, not for an already-running event environment.

![Run a stopped JupyterLab space](/static/images_prereq/sagemaker_studio_space_run.png)

![Open the JupyterLab space](/static/images_prereq/sagemaker_studio_space_open.png)

In an own-account environment without the workshop template, follow the [JupyterLab guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide.html) with your administrator to create or select an approved space. Such a space may not have the workshop's automatic repository checkout configured.

## 3. Locate the workshop notebooks

The workshop lifecycle configuration automatically clones the [source repository](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker) into `/home/sagemaker-user/generative-ai-on-amazon-sagemaker` when that destination is absent. It uses a shallow, sparse checkout containing only `workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai`, so repository-root files and sibling workshops are not downloaded into the file browser. The default source is `main`; your instructor may select another published branch or tag through `WorkshopRepositoryRef`.

In the JupyterLab file browser, double-click **generative-ai-on-amazon-sagemaker**, then **workshops**, then **fine-tune-nvidia-nemotron-3-sagemaker-ai**. Choose the directory for your selected lab and open `1-prepare-data.ipynb`.

```text
generative-ai-on-amazon-sagemaker/
└── workshops/
    └── fine-tune-nvidia-nemotron-3-sagemaker-ai/
        ├── 01-serverless-workshop/
        │   └── 1-prepare-data.ipynb
        ├── 02-smtj-workshop/
        │   └── 1-prepare-data.ipynb
        └── README.md
```

| Lab                                         | Directory below the workshop root |
| ------------------------------------------- | --------------------------------- |
| [Serverless customization](/02-serverless/) | `01-serverless-workshop/`         |
| [Training jobs](/03-training-jobs/)         | `02-smtj-workshop/`               |

The file-browser breadcrumb **/** is the JupyterLab root, not the operating system's root directory. The workshop `README.md` is inside the workshop folder. Existing checkouts are preserved on startup, so a previously used space may contain additional files or an earlier source revision. Startup does not update an existing clone or install Python requirements. If an expected notebook is missing, ask your instructor or administrator to verify the source revision before changing the checkout.

### If automatic checkout is unavailable

Open a terminal in JupyterLab using **File → New → Terminal**, or choose **Terminal** under **Other** in the Launcher.

![Open a terminal in JupyterLab using File → New → Terminal](/static/images_prereq/jupyterlab-open-terminal.png)

In the terminal, clone the repository:

```bash
git clone https://github.com/aws-samples/generative-ai-on-amazon-sagemaker.git
```

Then open the workshop folder in the file browser and select your lab. If the repository folder already exists, use that checkout instead of cloning again.

## 4. Select the Python kernel and install dependencies

Open `1-prepare-data.ipynb` from your selected lab directory and select **Python 3 (ipykernel)**. Keep the notebook in that directory so it can find its `requirements.txt`, configuration, and helper modules.

Run the notebook's installation cell:

```ipython
%pip install -r requirements.txt
```

Inspect the full cell output for errors and wait for the kernel to return to **Idle**. Restart the kernel if requested before running the remaining setup cells. Both labs specify `sagemaker>=3.16.0`; use the supplied requirements rather than installing a different SDK generation. This installation prepares the notebook environment, not the separate training or serving environments.

## 5. Verify the working directory and imports

After installing dependencies, you can use this local inspection cell to confirm that the notebook imports the intended configuration and helpers:

```python
from pathlib import Path
import sys
import config
import contractnli as C

print("working directory:", Path.cwd())
print("python:", sys.executable)
print("config:", config.__file__)
print("helper:", C.__file__)
print("model:", config.BASE_MODEL_ID)
```

Both imported files should come from the selected lab directory. The two directories contain helper modules with the same names but different behavior. If you opened the wrong directory, restart the kernel in the correct one rather than continuing with cached imports.

| Lab                      | Expected model configuration                                |
| ------------------------ | ----------------------------------------------------------- |
| Serverless customization | `huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16` |
| Training jobs            | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`                     |

## 6. Run the notebook's session setup

Run the setup cells in `1-prepare-data.ipynb` in order. They create a SageMaker session, resolve the default bucket and optional prefix, identify the execution role, and initialize clients. Check the displayed values against your intended account and Region:

```text
sagemaker role arn: <execution-role ARN>
sagemaker bucket: <bucket name>
sagemaker session region: <AWS Region>
```

The session can create a default S3 bucket if one is needed. Confirm that the displayed role, bucket, Region, and configured upload prefix belong to your workshop environment before continuing to data preparation.

## Troubleshooting

| Symptom                           | Check                                                                                     |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| Wrong model ID printed            | Verify the selected directory and imported file paths, then restart the kernel if needed. |
| `ModuleNotFoundError`             | Run the selected lab's requirements cell in the active kernel.                            |
| Missing SageMaker class or import | Check the installed SDK version and confirm the requirements cell completed successfully. |
| Role resolution fails             | Verify the Studio execution role or approved fallback role with your administrator.       |
| S3 access denied                  | Check the displayed role, bucket/prefix, and relevant object and KMS permissions.         |
| Model or tokenizer download fails | Check model access and the outbound network path.                                         |

Once the dependencies load and the session identifies the intended environment, continue to [Serverless customization](/02-serverless/) or [Training jobs](/03-training-jobs/).
