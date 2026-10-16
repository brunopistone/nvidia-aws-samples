---
title: "Fine-tuning"
weight: 2
---

## What you will learn

You will assemble a SageMaker Training job from source code, a container, compute configuration, and three input channels; interpret the model-specific BF16 LoRA recipe; follow how TRL loads, trains, and exports the model; and preserve an exact artifact handoff for deployment.

::alert[Open [02-smtj-workshop/2-fine-tune-llm.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/2-fine-tune-llm.ipynb). The executable configuration uses **one epoch on one single-GPU instance**, selected from five ordered Instance Preferences with `ml.g5.2xlarge` preferred. Older notebook paragraphs describing ten epochs, four GPUs, or four data channels do not describe the current code.]{type="info"}

## Understand the ModelTrainer

`ModelTrainer` is the SageMaker SDK object that submits the remote Training job. The job runs [scripts/train.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/scripts/train.py), which constructs TRL's `SFTTrainer` and applies PEFT adapters. Here you supply the Hugging Face model ID, container, dependency versions, script, configuration, and instance selection.

The configured base model is `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`. The recipe loads BF16 weights and adds LoRA; `load_in_4bit: false` means this run is **not QLoRA**. Installed dependencies such as bitsandbytes, DeepSpeed, or distributed helpers do not mean those execution paths are enabled.

### SFT is the objective; LoRA is the parameter-update method

![Completion-only SFT compares predicted answer tokens with expert JSON and updates LoRA adapters while base weights remain frozen.](/static/images/general/sft-contractnli.svg)

SFT learns to produce expert demonstrations under a given input. Training uses the correct preceding answer tokens when predicting each next token, often called teacher forcing. Serving has no gold answer and instead uses its own generated prefix. A low training loss therefore does not guarantee a complete or correct generated checklist. Hold out whole contracts and measure verdicts and citations after deployment.

This recipe masks prompt and padding positions so only completion tokens contribute to the loss. Masking does not remove the contract from context: the model still attends to that context when predicting the answer. Inspect full template-rendered lengths because truncation can remove supervised tokens or entire checklist entries. Validation loss uses the development split; the held-out test split is not a training input.

For an ordinary LoRA update, `W_updated = W + (alpha / rank) * B * A`. If `W` has `output_dim * input_dim` parameters, the two low-rank matrices contain `rank * (input_dim + output_dim)` trainable parameters. Base weights stay frozen. Rank controls update capacity, alpha controls scaling, and dropout regularizes the adapter path; none directly measures task quality. This reduces trainable parameters and optimizer state, but does not remove base weights, activations, or temporary tensors from memory. The configured rank is 32 and alpha is 64.

A merged export applies the learned update to the base weights so serving does not need a separate adapter. The merged artifact is larger than the adapter alone and must include compatible configuration and tokenizer files. Select target modules by their actual execution path: a module name that exists in the architecture is not sufficient if fused code bypasses its adapted forward method. The model-specific table below explains those choices.

## 1. Verify the inputs before provisioning compute

Run the notebook's session setup, then rebuild the paths from the same `DATA_PREFIX` used in data preparation:

```python
from config import BASE_MODEL_ID, DATA_PREFIX, TRAIN_JOB_PREFIX

input_path = f"{default_prefix}/{DATA_PREFIX}" if default_prefix else DATA_PREFIX
train_dataset_s3_path = f"s3://{bucket_name}/{input_path}/train/dataset.jsonl"
val_dataset_s3_path = f"s3://{bucket_name}/{input_path}/val/dataset.jsonl"
train_config_s3_path = f"s3://{bucket_name}/{input_path}/config/args.yaml"

for uri in (train_dataset_s3_path, val_dataset_s3_path):
    key = uri.split(f"{bucket_name}/", 1)[1]
    size = s3_client.head_object(Bucket=bucket_name, Key=key)["ContentLength"]
    print(uri, size)
```

An object-size response confirms existence and access, not schema correctness. Validate the conversational `prompt`, `completion`, and `chat_template_kwargs` fields from the previous lesson. These default keys can be overwritten by the serverless track, which uses a different schema.

| Input                                     | File inside the Training job             | Purpose                                |
| ----------------------------------------- | ---------------------------------------- | -------------------------------------- |
| `train`                                   | `/opt/ml/input/data/train/dataset.jsonl` | Optimization examples                  |
| `val`                                     | `/opt/ml/input/data/val/dataset.jsonl`   | Validation loss                        |
| `config`                                  | `/opt/ml/input/data/config/args.yaml`    | Script arguments and TRL configuration |
| Source bundle, not an `InputData` channel | Uploaded `./scripts` directory           | `train.py` and training requirements   |

No held-out test file belongs in these channels. There are three `InputData` objects in the executable notebook, and the source bundle is supplied separately.

## 2. Read the inline recipe

The notebook exports `model_id`, `mlflow_uri`, and `mlflow_experiment_name` into `os.environ`. Its subsequent `%%bash` cell writes `./args.yaml` using shell interpolation. A shell cell cannot see ordinary Python variables unless they have been exported into the process environment.

```python
import os

os.environ["model_id"] = BASE_MODEL_ID
os.environ["mlflow_uri"] = mlflow_app_arn
os.environ["mlflow_experiment_name"] = "nemotron-3-nano-4b-contractnli-sft"
```

The setup cell resolves a ready `mlflow_app_arn` before continuing. If `MLFLOW_APP_ARN` is configured, it reuses that exact app and waits; an invalid or inaccessible explicit ARN fails without replacement. Leaving it empty first reuses the uniquely matching tagged `workshop-mlflow-<stack UUID>` app, scoped to the notebook role, ahead of any `nemotron-contractnli-workshop` duplicate. Ambiguous matches stop setup and request an explicit ARN; pending apps are waited for, and failed apps are not replaced. Only if no stack app matches does setup reuse or create `nemotron-contractnli-workshop` in the current account/Region, using the notebook role and artifact bucket unless overrides are set. No manual ARN or stack update is required for discovery. See [MLflow setup behavior](/01-prerequisites/3-sagemaker/#7-reuse-or-create-the-workshop-mlflow-app). The following YAML reproduces the executable settings with comments condensed, the model variable shown as its configured value, and the MLflow ARN shown as a placeholder. It is a configuration reference for the notebook's existing write cell, not a separate `scripts/args.yaml` file. Run the notebook's full shell cell, inspect the expanded result for the actual app ARN, then run its upload cell.

```yaml
model_id: "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
mlflow_uri: "<resolved MLflow App ARN>"
mlflow_experiment_name: "nemotron-3-nano-4b-contractnli-sft"
mlflow_log_model: false
train_dataset_path: "/opt/ml/input/data/train/dataset.jsonl"
val_dataset_path: "/opt/ml/input/data/val/dataset.jsonl"
dataset_format: "prompt_completion"
attn_implementation: "flash_attention_2"
torch_dtype: "bfloat16"
cast_parameters_to_uniform_dtype: false
load_in_4bit: false
use_peft: true
merge_weights: true
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "up_proj"
  - "down_proj"
  - "in_proj"
pad_token: "<unk>"
eos_token: "<|im_end|>"
early_stopping: true
output_dir: "/opt/ml/model"
max_length: 8192
packing: false
completion_only_loss: true
num_train_epochs: 1
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
warmup_steps: 0.1
lr_scheduler_type: "cosine"
max_grad_norm: 1.0
weight_decay: 0.0
bf16: true
gradient_checkpointing: true
ddp_find_unused_parameters: false
logging_strategy: "steps"
logging_steps: 5
eval_strategy: "epoch"
save_strategy: "epoch"
save_total_limit: 1
seed: 42
```

`TrlParser((ScriptArguments, SFTConfig))` parses both the script-specific fields and the trainer configuration from this one file. Unknown fields or incompatible dependency versions can fail at startup; do not copy a configuration key from another version without checking it. In this source environment, `warmup_steps: 0.1` is intended as a ratio, while an integer represents a step count. Preserve the numeric type when editing YAML.

### Why these settings are model-specific

| Setting                                              | Rationale and checkpoint                                                                                                                                               |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Explicit adapter targets                             | Adapt attention projections, MLP projections, and Mamba input projections rather than indiscriminately targeting all linear layers                                     |
| No `out_proj` in the target list                     | The notebook documents a fused Mamba path reading `out_proj.weight` directly; wrapping the module does not necessarily put an adapter on the executed computation path |
| No `gate_proj` target                                | Do not assume another architecture's gated-MLP naming exists here                                                                                                      |
| `load_in_4bit: false`                                | Use the verified BF16 recipe rather than assuming an arbitrary quantized path is compatible                                                                            |
| `pad_token: "<unk>"` and `eos_token: "<\|im_end\|>"` | Keep padding distinct from the actual chat turn terminator and preserve stopping behavior                                                                              |
| `packing: false`                                     | Retain the selected prompt/completion training layout; packing is a separate change requiring mask validation                                                          |
| Explicit validation batch size of one                | Avoid accidentally evaluating with a larger default batch than fits the training configuration                                                                         |
| Gradient checkpointing                               | Trade recomputation for reduced stored activations; it does not guarantee that larger batches or sequences fit                                                         |

The script initially falls back to the tokenizer's EOS if no pad token exists. The recipe supplies the explicit pad/EOS overrides through `SFTConfig`; inspect the tokenizer used by TRL and the exported tokenizer rather than stopping at that early fallback. Padding-token identity matters to masking and termination, but the exact collator implementation should be checked rather than assuming all versions mask tokens in the same way.

The helper favors Transformers' native model implementation when it is available instead of forcing remote modeling code. The notebook deliberately leaves `trust_remote_code` unset and requests `flash_attention_2`. This is an implementation compatibility decision, not permission to execute arbitrary remote code without review.

### Effective batch and early stopping

The selected instance has one GPU. With one sample per device and four accumulation steps, the nominal effective batch is `1 GPU * 1 record * 4 accumulation steps = 4 records` per optimizer update, except for a final partial accumulation. `Torchrun()` remains configured, but does not make this a four-GPU job. If every one of 423 records contributes and final partial accumulation is kept, the simple arithmetic is `ceil(423 / 4) = 106` updates per epoch. Here `ceil` means round up to the next integer; use actual trainer logs for the realized count.

`early_stopping: true` adds `EarlyStoppingCallback(patience=3, threshold=0.01)` and selects the best model by decreasing `eval_loss`. Both save and evaluation strategies are `epoch`, satisfying the intended best-checkpoint workflow. However, a one-epoch run cannot establish a multi-epoch plateau with patience three. The current recipe does not automatically extend itself to ten epochs. Treat validation loss as a training diagnostic, not a substitute for verdict and evidence scoring.

## 3. Inspect the training environment

### Instance Preferences

The notebook supplies five candidates in this exact preference order:

| Order | Instance type    | GPU  | GPU memory | Host RAM |
| ----- | ---------------- | ---- | ---------- | -------- |
| 1     | `ml.g5.2xlarge`  | A10G | 24 GB      | 32 GB    |
| 2     | `ml.g6.2xlarge`  | L4   | 24 GB      | 32 GB    |
| 3     | `ml.g6e.2xlarge` | L40S | 48 GB      | 64 GB    |
| 4     | `ml.g5.xlarge`   | A10G | 24 GB      | 16 GB    |
| 5     | `ml.g6.xlarge`   | L4   | 24 GB      | 16 GB    |

SageMaker selects the first type with available capacity and launches one instance of that type, not one of every candidate. If none has capacity, this GPU job waits and retries subject to the service's pending-time limit. Preferences do not reserve capacity, guarantee availability, or choose the cheapest or fastest GPU. SageMaker selects by capacity, so the container and workload must support every candidate. See [Instance preference lists for training jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/train-instance-preferences.html).

Training quota must cover the requested count for **every** listed type; insufficient quota for any candidate rejects the request even if an earlier type could run. All five candidates have one GPU, so fallback preserves the nominal effective batch of four without changing the recipe. GPU architecture, memory, host RAM, runtime, and price can still differ. The notebook's timing example on `ml.g5.2xlarge` is a reference run, not a duration promise for the alternatives.

The notebook resolves a SageMaker PyTorch training image for the Region using the first preference. The same GPU image is used for every candidate; this lookup does not report which type SageMaker actually selects:

```python
from sagemaker.core import image_uris

instance_types = [
    "ml.g5.2xlarge",
    "ml.g6.2xlarge",
    "ml.g6e.2xlarge",
    "ml.g5.xlarge",
    "ml.g6.xlarge",
]
instance_count = 1
image_uri = image_uris.retrieve(
    framework="pytorch",
    region=region,
    version="2.8.0",
    instance_type=instance_types[0],
    image_scope="training",
)
print(image_uri)
```

The source bundle installs its own [scripts/requirements.txt](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/scripts/requirements.txt). Current pins include Transformers `5.15.1`, TRL `1.10.0`, PEFT `0.20.0`, Accelerate `1.14.0`, datasets `5.0.1`, and SageMaker `3.21.0`. The Mamba and causal-convolution wheels are explicitly built for the named Python/PyTorch/CUDA ABI, and Triton is pinned to `3.4.0`. Changing the base image without reviewing those binary dependencies can fail before training begins.

Notebook-side package installation does not update the training container, and changing training dependencies does not update the later serving image. Record each environment separately. The current serving notebook's image discussion is not proof of a compatible Transformers version inside its actual image.

## 4. Configure the SageMaker job

```python
from sagemaker.core.shapes import InstancePreference
from sagemaker.core.training.configs import (
    CheckpointConfig, Compute, OutputDataConfig, SourceCode, StoppingCondition,
)
from sagemaker.train.distributed import Torchrun
from sagemaker.train.model_trainer import ModelTrainer

source_code = SourceCode(
    source_dir="./scripts",
    requirements="requirements.txt",
    entry_script="train.py",
)
KEEP_ALIVE_SECONDS = 0
compute_configs = Compute(
    instance_preferences=[InstancePreference(instance_type=t) for t in instance_types],
    instance_count=instance_count,
    keep_alive_period_in_seconds=KEEP_ALIVE_SECONDS,
)
output_path = (f"s3://{bucket_name}/{default_prefix}/{TRAIN_JOB_PREFIX}"
               if default_prefix else f"s3://{bucket_name}/{TRAIN_JOB_PREFIX}")
runtime_environment = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

model_trainer = ModelTrainer(
    training_image=image_uri,
    source_code=source_code,
    environment=runtime_environment,
    base_job_name=TRAIN_JOB_PREFIX,
    compute=compute_configs,
    distributed=Torchrun(),
    stopping_condition=StoppingCondition(max_runtime_in_seconds=7200),
    hyperparameters={"config": "/opt/ml/input/data/config/args.yaml"},
    output_data_config=OutputDataConfig(s3_output_path=output_path),
    checkpoint_config=CheckpointConfig(
        s3_uri=output_path + "/checkpoint", local_path="/opt/ml/checkpoints",
    ),
    role=role,
    sagemaker_session=sess,
)
```

The allocator option is placed in the environment so it is present when PyTorch initializes CUDA. It targets allocation behavior, not total available memory; it cannot make an inherently oversized workload fit. The two-hour stopping condition is a maximum runtime setting, not a promised training duration.

::alert[The current notebook sets `KEEP_ALIVE_SECONDS = 0`, requesting no warm-pool retention after training. A nonzero value is an explicit opt-in to potentially billed retained compute and requires a separate warm-pool quota. Set the intended value **before** constructing and submitting the job. Changing the variable after submission does not update an existing pool; inspect earlier jobs separately.]{type="warning"}

### Checkpoint synchronization is not checkpoint placement

`CheckpointConfig` synchronizes `/opt/ml/checkpoints` to the configured S3 location. The inline YAML does **not** set the script's `checkpoint_dir`, and `use_checkpoints` remains false by default. In `train.py`, only a non-empty `checkpoint_dir` redirects TRL's output directory for intermediate checkpoints; resuming additionally requires `use_checkpoints` and an existing checkpoint. As written, epoch checkpoints follow the trainer's `/opt/ml/model` output directory rather than automatically appearing in the synchronized directory. Do not claim that this configuration provides interruption recovery without separately wiring and testing those settings.

## 5. Upload configuration and submit three channels

Run the existing `%%bash` write cell first. Then the notebook uploads `args.yaml` to `config/args.yaml` and removes the local temporary file. Save a copy of the exact expanded configuration with your experiment records before cleanup if you need reproducibility.

```python
from sagemaker.core.training.configs import InputData

data = [
    InputData(channel_name="train", data_source=train_dataset_s3_path),
    InputData(channel_name="val", data_source=val_dataset_s3_path),
    InputData(channel_name="config", data_source=train_config_s3_path),
]
model_trainer.train(input_data_config=data, wait=False, logs=False)
training_job = model_trainer._latest_training_job
if training_job is None:
    raise RuntimeError("No TrainingJob was created. Check the SDK version and session.")
TRAINING_JOB_NAME = training_job.training_job_name
print(f"Launched: {TRAINING_JOB_NAME}")
```

The current SDK path stores the created resource on `_latest_training_job`; `train()` does not return that resource. This is version-sensitive private SDK state, so use the supplied notebook/dependency versions and retain the printed name. The notebook's **Follow the run** cell prints job details and CloudWatch links plus a five-minute pre-signed MLflow App URL. Rerun that cell for a fresh UI link if it expires; do not save the signed URL as a public reference. The subsequent wait cell polls primary/secondary status every 60 seconds and raises for `Failed` or `Stopped` rather than treating any terminal state as success.

After polling reaches a terminal state, that cell reports the actual selection from `DescribeTrainingJob`, using the following lines:

```python
selected = job["ResourceConfig"].get("SelectedInstanceType", "not selected yet")
print(f"\ninstance type: {selected}")
```

`SelectedInstanceType` is absent until selection occurs; the fallback text is not an instance name. Do not infer the actual type from `instance_types[0]` or the image lookup. Retain the reported type with the job record and use its regional SageMaker training rate for cost analysis. The configured count remains one regardless of which candidate is selected.

The method returns before the remote work is complete. Save the exact submitted job name rather than relying only on a latest-match lookup later. Use the wait time for these progress activities:

1. In **SageMaker AI Studio**, open **Jobs > Training** and select the submitted job. Alternatively, open the [SageMaker AI console](https://console.aws.amazon.com/sagemaker/) in the same Region and choose **Training > Training jobs**. Compare the exact name, input channels, image, ordered instance preferences, and count of one with your configuration. In the **Instance type preferences** table, record the type marked **Selected**, or **Pending selection** while capacity is unresolved. Note both primary and secondary status.
2. From the job details, follow **View logs** to [CloudWatch](https://console.aws.amazon.com/cloudwatch/). If that link is unavailable, open **Logs > Log groups > `/aws/sagemaker/TrainingJobs`** and filter streams by the exact job name. Set the time window to this run. Find dependency installation, dataset loading, model initialization, optimizer updates, validation, and export rather than interpreting a quiet notebook as a stalled job.
3. Before metric points appear, trace one train-channel URI back to data preparation and identify the YAML fields controlling batch size, completion-only loss, and the token cap. During training, compare actual optimizer steps with the nominal `ceil(423 / 4)` calculation. At epoch end, inspect validation loss, then confirm merging and upload finish before calling the job complete.
4. Open the configured MLflow App as described below. Compare training and validation metrics, recording the run ID and corresponding SageMaker job name. Missing startup metrics may simply precede the training loop; use CloudWatch to distinguish healthy installation from failure.

**Checkpoint:** record the current stage and one piece of evidence for it. A completed polling cell is not proof of `Completed`, and a completed training loop is not proof that export succeeded. If you interrupt monitoring, the remote job continues.

## 6. Follow the script's actual execution path

| Stage in `train.py`     | What this recipe does                                                               | Evidence to inspect                                       |
| ----------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Parse and load          | Read YAML with `TrlParser`; load JSONL train/validation files                       | Resolved settings, sample counts, column names            |
| Model/tokenizer setup   | Load the configured causal LM in BF16 and resolve native/remote implementation      | Loading class, attention backend, dependency errors       |
| Dataset preparation     | Pass prompt/completion columns through to TRL                                       | Correct columns and tokenization/masking behavior         |
| PEFT setup              | Construct `LoraConfig` with the explicit target list and wrap the model             | Printed trainable-parameter summary and healthy gradients |
| Trainer creation        | Pass model, datasets, tokenizer, and callbacks to TRL `SFTTrainer`                  | Loss, learning rate, epoch, and validation logs           |
| Save and merge          | Save an adapter temporarily, reload base plus adapter, merge, and serialize weights | Export-stage success, not only the last training step     |
| Final artifact handling | Save tokenizer and align generation configuration                                   | Loadable full checkpoint under `/opt/ml/model`            |

The training script calls the validation dataset `test_ds` internally, but it loads that variable from `val_dataset_path`. It is not the held-out local `./tmp/test.jsonl`. This naming should not lead you to include test data in training diagnostics.

### Open the configured MLflow App and inspect this run

MLflow is enabled by the resolved app ARN and non-empty experiment name in `args.yaml`. The script sets its tracking URI and experiment, includes `mlflow` in `report_to`, and logs training metrics to that app. In **SageMaker AI Studio**, choose the **MLflow** application, find the **MLflow App** matching the resolved `mlflow_app_arn` printed during setup, and open its UI. Select **Experiments > `nemotron-3-nano-4b-contractnli-sft`**, then the run whose start time corresponds to your job. Record its run ID and inspect parameters, training loss, learning rate, epoch/step, and validation loss when available. Metric names and availability follow the installed trainer and logging cadence; do not assume every plot exists before training starts.

Use the standalone app resolved during setup, whether it was precreated, discovered by the workshop name, or newly created. Save its ARN in `config.py` for explicit reuse. Do not create a legacy MLflow Tracking Server or a second Studio domain. The app itself requires no domain association; Studio is a convenient launch surface, not the app's lifecycle owner. See [MLflow App setup](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-app-setup.html). If the app is not visible, compare the printed ARN/account/Region and permissions with the instructor instead of provisioning a substitute. Never share a temporary signed UI URL as a durable workshop link.

The training container installs `mlflow==3.16.1` and `sagemaker-mlflow==0.5.0`. Full model logging is separately controlled by `mlflow_log_model`, which defaults to false; experiment metrics do not require logging another copy of the merged weights. MLflow tracking does not change the deployment handoff: notebook 3 uses `model.tar.gz` in S3. The CloudFormation-provisioned app disables automatic registration into SageMaker Model Registry; reused apps retain their existing settings, so verify those separately rather than assuming setup changes them.

### Export and generation configuration

With `merge_weights: true`, the text-only path saves the adapter to a temporary directory and uses `AutoPeftModelForCausalLM` followed by `merge_and_unload`. The export writes safe-serialization weight shards with a configured maximum shard size of `2GB`, the tokenizer, and generation configuration. SageMaker then packages the output directory as `model.tar.gz`.

The script includes compatibility handling for weight conversion and tied-weight metadata. It also resets sampling-only generation parameters when `do_sample` is not true before saving, and aligns exported EOS/pad IDs with the tokenizer. These are serialization and stopping-consistency safeguards, not proof that sampled generation always produces invalid JSON. The serving request later controls decoding explicitly.

## 7. Validate the handoff

Wait for the Training job to reach `Completed`. A failed merge or upload can occur after the training loop has finished. The job, rather than Model Registry, is the source of the artifact URI. The following read-only monitoring excerpt uses the same `describe_training_job` API as deployment; set `TRAINING_JOB_NAME` to the exact submitted name first.

```python
job = sm_client.describe_training_job(TrainingJobName=TRAINING_JOB_NAME)
print(job["TrainingJobStatus"])
assert job["TrainingJobStatus"] == "Completed"
model_data_url = job["ModelArtifacts"]["S3ModelArtifacts"]
print(model_data_url)
```

Expected output is a status and an S3 URI ending in the job's `output/model.tar.gz`, not a Model Package ARN. Keep job name, input URIs, expanded YAML, image URI, source revision, dependency versions, and artifact URI together. This recipe registers neither an AI Registry dataset nor a SageMaker Model Package.

| Failure or uncertainty            | Targeted check                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| YAML parser aborts                | Check unknown fields, types, duplicate argument names, and actual TRL/Transformers versions |
| Dependency installation fails     | Review wheel Python/PyTorch/CUDA ABI and pinned Triton compatibility                        |
| Training or first validation OOM  | Check the phase, both per-device batch sizes, token lengths, and allocator logs             |
| Adapter receives no gradient      | Inspect target-module execution paths; do not substitute `all-linear` blindly               |
| Job fails after the final epoch   | Inspect merge and generation-config serialization errors                                    |
| No resumable S3 checkpoint        | Check script `checkpoint_dir`/`use_checkpoints`, not just `CheckpointConfig`                |
| Unexpected post-job compute usage | Inspect warm-pool state and retention setting                                               |

Continue to [Deployment](/03-training-jobs/03-deployment/) with **`3-deployment.ipynb`**, then [Evaluation](/03-training-jobs/04-evaluation/) with **`4-evaluation.ipynb`**. The current source filenames match this execution order. Keep the endpoint running only while collecting fresh predictions, then use the evaluation notebook's cleanup section and the shared [Clean Up](/04-cleanup/) checklist.
