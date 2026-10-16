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

| Lab                                                         | Directory below the workshop root                 |
| ----------------------------------------------------------- | ------------------------------------------------- |
| [SageMaker Training jobs: 4B](/03-training-jobs/)           | `02-smtj-workshop/` (primary journey)             |
| [Optional: Serverless model customization](/02-serverless/) | `01-serverless-workshop/` (alternative extension) |

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

Inspect the full cell output for errors and wait for the kernel to return to **Idle**, then restart the kernel before running setup. Both requirements files load the shared [constraints-notebook.txt](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/constraints-notebook.txt), which pins `sagemaker==3.21.0`, `sagemaker-core==2.21.0`, and train/serve/mlops modules at `1.21.0`, alongside compatible notebook dependency versions. Keep the parent workshop directory: installing or copying a lab directory alone loses this constraints file and the shared MLflow helper. The SDK metapackage version alone does not establish module compatibility. This installation prepares the notebook environment, not the separate training or serving environments; do not install `scripts/requirements.txt` or its CUDA wheels in JupyterLab.

The source README records successful wheel-only pip resolver dry runs targeting Linux x86_64 / CPython 3.12. These are not proof of a working Studio runtime or compatibility with every preinstalled extension, and cross-platform resolution does not fully emulate native Linux environment markers. In the actual Studio kernel, run `%pip check` after installation and restart, then smoke-test the notebook imports before a paid job. A native Linux install/import check and separate training-DLC smoke test remain necessary.

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

| Lab                                      | Expected model configuration                                |
| ---------------------------------------- | ----------------------------------------------------------- |
| SageMaker Training jobs: 4B              | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`                     |
| Optional: Serverless model customization | `huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16` |

## 6. Run the notebook's session setup

Run the setup cells in `1-prepare-data.ipynb` in order. They create a SageMaker session, resolve the default bucket and optional prefix, identify the execution role, and initialize clients. Check the displayed values against your intended account and Region:

```text
sagemaker role arn: <execution-role ARN>
sagemaker bucket: <bucket name>
sagemaker session region: <AWS Region>
```

The session can create a default S3 bucket if one is needed. Confirm that the displayed role, bucket, Region, and configured upload prefix belong to your workshop environment before continuing to data preparation.

## 7. Reuse or create the workshop MLflow App

A configured ARN is **optional**. With an empty ARN, both journeys first discover the precreated workshop app automatically. To override discovery, copy the stack's `MlflowAppArn` output into `MLFLOW_APP_ARN` in the selected journey's `config.py`, or use an administrator-approved app ARN in the same Region. Use an `mlflow-app/...` ARN, not a legacy `mlflow-tracking-server/...` ARN or a UI URL. An explicit ARN always takes precedence: invalid, inaccessible, missing, or failed apps stop setup without silently creating a replacement. An app still creating or updating is polled until ready or timed out.

With `MLFLOW_APP_ARN = ""`, the shared [workshop_mlflow.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/workshop_mlflow.py) first lists apps and reads `ListTags` for `workshop-mlflow-*` candidates. A match requires the template's `WorkshopStackId` tag to contain a CloudFormation stack ARN in the same account/Region, whose UUID exactly matches the app-name suffix. A supplied role must match the app role; without a role, a supplied artifact URI must match (ignoring trailing slashes). Role takes precedence because the notebook default bucket can differ from the stack bucket. Without either scope, only a unique tag/name match is accepted. Multiple matches stop and request an explicit ARN. This app wins over a previously auto-created `nemotron-contractnli-workshop` duplicate. Pending apps are waited for; failed or unsupported states never trigger replacement. Listing is bounded to 100 pages per operation, and incomplete scans or access errors stop setup rather than creating an app.

Only if no stack app matches does setup reuse the exact `MLFLOW_APP_NAME`, then use these fallback paths if that app is also absent:

| Journey                                  | Automatic behavior when no ARN is configured                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SageMaker Training jobs: 4B              | List apps in the current account/Region and reuse the exact `MLFLOW_APP_NAME`, default `nemotron-contractnli-workshop`, or create it. Creation uses the notebook execution role and `s3://<notebook-bucket>/mlflow-artifacts`; `MLFLOW_ROLE_ARN` and `MLFLOW_ARTIFACT_STORE_URI` optionally override them. Existing app settings are not changed. Multiple same-name apps require an explicit ARN; a failed same-name app is not replaced. |
| Optional: Serverless model customization | Use the pinned SDK's native discovery/create path: check available domain/account defaults and ready apps, or create a timestamp-named `finetune-mlflow-*` app. No domain is required. Native creation uses the SDK role default and `s3://sagemaker-<region>-<account>/mlflow-artifacts`, with bucket/prefix creation if needed. Set both role/artifact overrides to use the deterministic named workshop path instead.                   |

Both creation paths use `AccountDefaultStatus="DISABLED"`, assign no domain defaults, and do not update existing apps. No Studio domain or legacy Tracking Server is required for the standalone MLflow App; Studio is only a convenient launch surface. CloudFormation precreation is a startup optimization, not a prerequisite. Existing stacks already carry the tag/name combination used for discovery; no stack update is needed.

The shared helper confirms readiness with `DescribeMlflowApp(Arn=...)`, polling creating/updating apps every 15 seconds for up to `MLFLOW_READY_TIMEOUT_SECONDS` (900 by default). The native SDK creation path first has its own wait of up to 600 seconds. Save the printed ARN in `config.py` to reconnect after a timeout and to keep training and evaluation on the same app, especially in shared accounts. Interrupting setup does not cancel app provisioning. SDK discovery/create errors stop setup rather than starting untracked training.

Serverless consumers receive the resolved ARN explicitly; the Training-job notebook passes it as `mlflow_uri` in its inline `args.yaml`. Confirm [IAM and artifact-store permissions](/01-prerequisites/2-account/) for your chosen path before setup. Existing Studio checkouts remain untouched by startup. Local edits on another machine do not update Studio: transfer the revised `workshop_mlflow.py` and both track `config.py` files, or update the checkout to the published revision after preserving local work. Restart the notebook kernel, rerun setup with an empty ARN, and verify the printed ARN identifies the stack app before starting training. No CloudFormation update or deletion of an existing duplicate is required.

## Troubleshooting

| Symptom                           | Check                                                                                     |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| Wrong model ID printed            | Verify the selected directory and imported file paths, then restart the kernel if needed. |
| `ModuleNotFoundError`             | Run the selected lab's requirements cell in the active kernel.                            |
| Missing SageMaker class or import | Check the installed SDK version and confirm the requirements cell completed successfully. |
| Role resolution fails             | Verify the Studio execution role or approved fallback role with your administrator.       |
| S3 access denied                  | Check the displayed role, bucket/prefix, and relevant object and KMS permissions.         |
| Model or tokenizer download fails | Check model access and the outbound network path.                                         |

Once the dependencies load and the session identifies the intended environment, continue to [SageMaker Training jobs: 4B](/03-training-jobs/). The [optional Serverless model customization](/02-serverless/) extension is an alternative, not a prerequisite.
