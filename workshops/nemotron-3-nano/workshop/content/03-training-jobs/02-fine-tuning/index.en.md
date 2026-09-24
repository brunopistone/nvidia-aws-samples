---
title: "Fine-tuning"
weight: 2
---

## What you will learn

You will assemble a SageMaker Training job from source code, a container, compute configuration, and three input channels; interpret the model-specific BF16 LoRA recipe; follow how TRL loads, trains, and exports the model; and preserve an exact artifact handoff for deployment.

::alert[Open [02-smtj-workshop/2-fine-tune-llm.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/2-fine-tune-llm.ipynb). The executable configuration currently uses **one epoch on one `ml.g5.2xlarge`**. Older notebook paragraphs describing ten epochs, four GPUs, or four data channels do not describe the current code.]{type="info"}

## Understand the ModelTrainer

`ModelTrainer` is the SageMaker SDK object that submits the remote Training job. The job runs [scripts/train.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/scripts/train.py), which constructs TRL's `SFTTrainer` and applies PEFT adapters. Here you supply the Hugging Face model ID, container, dependency versions, script, configuration, and instance selection.

The configured base model is `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`. The recipe loads BF16 weights and adds LoRA; `load_in_4bit: false` means this run is **not QLoRA**. Installed dependencies such as bitsandbytes, DeepSpeed, or distributed helpers do not mean those execution paths are enabled.

For an ordinary LoRA update, \(W' = W + (\alpha/r)BA\). Training learns the low-rank matrices while freezing the base weight matrix. This reduces trainable parameters and optimizer state, but does not remove the need to hold the base model and training activations. A merged export applies the learned update to the base weights so serving does not have to load an adapter separately.

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
os.environ["mlflow_uri"] = ""
os.environ["mlflow_experiment_name"] = "nemotron-3-nano-4b-contractnli-sft"
```

The following YAML reproduces the executable settings, with comments condensed and the model variable shown as its configured value. It is a configuration reference for the notebook's existing write cell, not a separate `scripts/args.yaml` file. Run the notebook's full shell cell, inspect the expanded result, then run its upload cell.

```yaml
model_id: "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
mlflow_uri: ""
mlflow_experiment_name: "nemotron-3-nano-4b-contractnli-sft"
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

The selected instance has one GPU. With one sample per device and four accumulation steps, the nominal effective batch is \(1\times1\times4=4\) records per optimizer update, except for a final partial accumulation. `Torchrun()` remains configured, but does not make this a four-GPU job. If every one of 423 records contributes and final partial accumulation is kept, the simple batch arithmetic is about \(\lceil423/4\rceil=106\) updates per epoch; use actual trainer logs for the realized count.

`early_stopping: true` adds `EarlyStoppingCallback(patience=3, threshold=0.01)` and selects the best model by decreasing `eval_loss`. Both save and evaluation strategies are `epoch`, satisfying the intended best-checkpoint workflow. However, a one-epoch run cannot establish a multi-epoch plateau with patience three. The current recipe does not automatically extend itself to ten epochs. Treat validation loss as a training diagnostic, not a substitute for verdict and evidence scoring.

## 3. Inspect the training environment

The notebook resolves a SageMaker PyTorch training image for the selected Region and instance type:

```python
from sagemaker.core import image_uris

instance_type = "ml.g5.2xlarge"
instance_count = 1
image_uri = image_uris.retrieve(
    framework="pytorch",
    region=region,
    version="2.8.0",
    instance_type=instance_type,
    image_scope="training",
)
print(image_uri)
```

The source bundle installs its own [scripts/requirements.txt](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/scripts/requirements.txt). Current pins include Transformers `5.15.1`, TRL `1.10.0`, PEFT `0.20.0`, Accelerate `1.14.0`, datasets `5.0.1`, and SageMaker `3.21.0`. The Mamba and causal-convolution wheels are explicitly built for the named Python/PyTorch/CUDA ABI, and Triton is pinned to `3.4.0`. Changing the base image without reviewing those binary dependencies can fail before training begins.

Notebook-side package installation does not update the training container, and changing training dependencies does not update the later serving image. Record each environment separately. The current serving notebook's image discussion is not proof of a compatible Transformers version inside its actual image.

## 4. Configure the SageMaker job

```python
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
KEEP_ALIVE_SECONDS = 1800
compute_configs = Compute(
    instance_type=instance_type,
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

::alert[The notebook retains training compute for `1800` seconds after the job via a warm pool. That can incur charges after training completes and needs the applicable quota. For a one-shot workshop run, set `KEEP_ALIVE_SECONDS = 0` **before** constructing and submitting the job if you do not intend to retain a pool. Changing the variable after submission does not update an existing pool.]{type="warning"}

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
model_trainer.train(input_data_config=data, wait=False)
```

The method returns before the remote work is complete. Save the exact submitted job name from the notebook output or Studio job listing, rather than relying only on a latest-match lookup later. Monitor Studio **Jobs**, **Training** and CloudWatch for installation, dataset loading, model initialization, training/evaluation, and export stages.

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

MLflow is disabled by the notebook's empty `mlflow_uri`. The script enables it only with a non-empty URI and experiment name; it sets `report_to` based on configured tracking integrations. Do not expect an MLflow dashboard from the default settings. CloudWatch and the trainer's saved metrics remain the places to inspect the default run.

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
