import re

# Base model for this lab.
#
# A Hugging Face model id, not a JumpStart one: this lab customizes the model with a
# SageMaker Training job running `scripts/train.py` (TRL SFTTrainer + PEFT LoRA), so the
# weights are pulled from the Hub inside the job rather than resolved from JumpStart.
#
# Nemotron 3 Nano 4B is not offered for SageMaker serverless customization, which is why
# this lab exists alongside the 30B-A3B serverless one: same dataset, same task, same
# four steps, different training backend.
BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

# A Hugging Face id is not a legal SageMaker resource name: every job, model, endpoint and
# endpoint config has to match
#
#     ([a-zA-Z0-9]([a-zA-Z0-9-]){0,62})(?<!-)
#
# so the `nvidia/` org prefix alone makes `BASE_MODEL_ID` unusable - `ValidationException:
# Member must satisfy regular expression pattern`. Drop the org and collapse anything that
# is not alphanumeric into a hyphen, once, here, so every notebook derives the same names.
MODEL_SLUG = re.sub(r"[^a-zA-Z0-9]+", "-", BASE_MODEL_ID.split("/")[-1]).strip("-")

# Fixed dataset / resource names used across the notebooks
DATASET_PREFIX = "contractnli-nda-review"

# Where notebook 1 writes the JSONL splits and notebook 2 reads them from. A Training job
# takes S3 URIs through `InputData` channels, so the splits stay plain S3 objects rather
# than AI Registry `DataSet` entries - that registry belongs to the serverless
# customization flow and has nothing to resolve a Training job's channels against.
DATA_PREFIX = f"datasets/{DATASET_PREFIX}"

# Base name for the training job. SageMaker appends a timestamp, so the job that runs is
# `<TRAIN_JOB_PREFIX>-<timestamp>` and its artifacts land under
# `s3://<bucket>/[<prefix>/]<TRAIN_JOB_PREFIX>/<full-job-name>/output/model.tar.gz`.
#
# Derived once, here, because notebook 2 uses it as `base_job_name` and notebook 4 uses it
# to find the last completed job. In the reference workshop this interpolation is written
# out in both notebooks, so editing one and not the other silently deploys the wrong run.
TRAIN_JOB_PREFIX = f"train-{MODEL_SLUG}-sft"
