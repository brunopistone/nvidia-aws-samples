# Base model for this lab.
#
# Must be a JumpStart model that (a) supports serverless customization and
# (b) can be served by the SageMaker LMI/DJL container and imported into
# Amazon Bedrock Custom Model Import.
#
# Verified end-to-end in this lab: huggingface-reasoning-NVIDIA-Nemotron-3-Nano-30B
#
# Known NOT to work end-to-end: huggingface-vlm-NVIDIA-Nemotron-3-Nano-30B3-5-4b (NVIDIA-Nemotron-3-Nano-30B3.5). Training
# succeeds, but no current LMI container recognises model type `NVIDIA-Nemotron-3-Nano-30B3_5`, so
# notebook 4 fails.
BASE_MODEL_ID = "huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16"

# Fixed dataset / resource names used across the notebooks
DATASET_PREFIX = "contractnli-nda-review"
