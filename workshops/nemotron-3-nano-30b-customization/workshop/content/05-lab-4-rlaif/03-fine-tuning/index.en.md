---
title: "Fine-Tuning with RLAIF"
weight: 3
---

## Launch Serverless RLAIF Training

::alert[📒 Open the notebook **`lab-4-reinforcement-learning-from-ai-feedback/3-fine-tune-model.ipynb`**]

### Configure the RLAIFTrainer

The `RLAIFTrainer` handles serverless RLAIF training with GRPO (Group Relative Policy Optimization):

```python
from sagemaker.train.rlaif_trainer import RLAIFTrainer
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.evaluator import Evaluator

training_dataset = DataSet.get(name=f"{project_prefix}-train")
val_dataset = DataSet.get(name=f"{project_prefix}-eval")

reward_model_id = "openai.gpt-oss-120b-1:0"
reward_prompt = Evaluator.get(name=f"{project_prefix}-reward-prompt")

# Keep base_job_name short so the "-YYYYMMDDHHMMSS" timestamp SageMaker appends
# survives the 63-char truncation and each run gets a unique job name.
base_job_name = f"{project_prefix}-{base_model_shortname}"[: 63 - 15].rstrip("-")

trainer = RLAIFTrainer(
    model=base_model_id,
    model_package_group=model_package_group_name,
    reward_model_id=reward_model_id,
    reward_prompt=reward_prompt.arn,
    training_dataset=training_dataset,
    validation_dataset=val_dataset,
    s3_output_path=s3_output_path,
    base_job_name=base_job_name,
    sagemaker_session=sess,
    role=role,
    accept_eula=True
)
```

### Key Parameters

**Model Configuration:**

- `model`: Base model JumpStart ID (`huggingface-llm-qwen2-5-7b-instruct`)
- `model_package_group`: Name for the model package group to store fine-tuned model
- `reward_model_id`: AI judge model ID (`openai.gpt-oss-120b-1:0`)
- `reward_prompt`: ARN of the registered reward prompt evaluator

**Training Configuration:**

- `training_dataset`: SageMaker AI Dataset for training
- `validation_dataset`: SageMaker AI Dataset used to validate during training
- `s3_output_path`: S3 location for training outputs
- `base_job_name`: Short job-name prefix (kept short so the timestamp SageMaker appends survives the 63-char limit)
- `sagemaker_session`: SageMaker session object
- `role`: IAM role for training job
- `accept_eula`: Accept model license agreement

### Inspect and Configure Hyperparameters

We set the number of epochs to 1 so training finishes in a reasonable timeframe for this workshop. All the other hyperparameters can be left at their defaults:

```python
from rich.pretty import pprint

trainer.hyperparameters.max_epochs = 1
pprint(trainer.hyperparameters.to_dict())
```

### Start Training

Launch the RLAIF training job. Setting `wait=False` runs it asynchronously so you can monitor progress separately without blocking the notebook:

```python
training_job = trainer.train(wait=False)
print(f"Created training job: {training_job.training_job_name}")
```

The training job runs serverlessly - no infrastructure management required. SageMaker AI automatically:

- Provisions compute resources
- Generates candidate responses for each prompt
- Calls the reward model to score responses
- Computes advantages using GRPO
- Updates model parameters
- Scales down when complete

You can monitor the job's progress from the SageMaker AI console under **Training** > **Training jobs**. When it completes, the fine-tuned model is automatically registered as a new version in the Model Package Group you configured above (`model_package_group_name`), ready for evaluation and deployment.

:::alert{header="Training Time" type="info"}
RLAIF training typically takes 45-60 minutes for this dataset size. The reward model is called for each candidate response, which adds overhead compared to SFT or DPO.
:::

:::alert{header="Short on time?" type="warning"}
If you don't want to wait for training to complete, you can use notebook **`3b-import-model-from-s3.ipynb`** to download a pre-trained checkpoint from Hugging Face, upload it to your S3 bucket, and register it as a Model Package directly in the SageMaker Model Registry. This lets you skip ahead to evaluation and deployment immediately.
:::

---

Once training is complete, you're ready to evaluate the fine-tuned model.
