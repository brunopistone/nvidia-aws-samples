---
title: "Reward Model Configuration"
weight: 2
---

## About the Reward Model

In RLAIF, the **reward model** is an AI judge that evaluates model responses during training. Instead of training a separate reward model from human preferences (as in RLHF), we use a powerful foundation model with a carefully crafted prompt to score responses.

---

## Configure the Reward Model

::alert[📒 Open the notebook **`lab-4-reinforcement-learning-from-ai-feedback/2-prepare-reward-model.ipynb`**]

### The Reward Prompt

The reward prompt defines how the AI judge evaluates responses. Our prompt instructs the judge to:

- Score model responses from 0 (clearly artificial) to 1 (indistinguishable from human)
- Focus on naturalness, tone, and style
- De-emphasize completeness and length to avoid rewarding verbose outputs
- Evaluate independently — no ground truth reference is used during training

```python
reward_prompt = (
    "You are an expert human evaluator assessing how human-like and natural "
    "a text response sounds. Given the original question and an LLM-generated "
    "response, rate how human-like the response is based on the following aspects:\n\n"
    "- Naturalness and fluidity of language\n"
    "- Appropriateness and consistency of tone\n"
    "- Stylistic nuances and variability typical of human writing\n"
    "- Coherence and logical flow without artificial phrasing or repetitive structures\n"
    "- Avoidance of common LLM patterns (e.g., excessive hedging, bullet-point lists, "
    "overly formal or generic phrasing)\n\n"
    "Instructions:\n"
    "- Read the question carefully.\n"
    "- Evaluate whether the response sounds like it was written by a knowledgeable human "
    "in a natural conversation.\n"
    "- Ignore spelling errors.\n"
    "- Don't over-index on completeness and length of the answer.\n"
    "- Assign a score from 0 to 1 representing how human-like the model's reply sounds:\n"
    "   - 1 means indistinguishable from a natural human response.\n"
    "   - 0 means obviously artificial, unnatural, or inconsistent with human language norms.\n\n"
    "Output Format:\n"
    "Return a JSON object with EXACTLY the following structure (no extra text):\n"
    '{\n  "score": <numeric>,\n  "reasoning": "<brief explanation highlighting key '
    'differences or strengths>"\n}\n\n'
    "Prompt: {{ prompt }}\n"
    "LLM-generated response: {{ response }}"
)
```

### Upload Reward Prompt to S3

The reward prompt is saved to S3:

```python
reward_prompt_path = f"{project_prefix}_reward_prompt.txt"
with open(reward_prompt_path, "w") as f:
    f.write(reward_prompt)

s3_client = boto3.client('s3')
bucket_name = sess.default_bucket()

file_name = f"{project_prefix}_reward_prompt.txt"
prefix_key = f"{project_prefix}/{file_name}"
s3_client.upload_file(file_name, bucket_name, prefix_key)

reward_prompt_uri = f"s3://{bucket_name}/{prefix_key}"
```

### Register the Reward Prompt

The reward prompt is registered as a reusable asset in the SageMaker AI Registry:

```python
from sagemaker.ai_registry.evaluator import Evaluator
from sagemaker.ai_registry.air_constants import REWARD_PROMPT

reward_prompt = Evaluator.create(
    name=f"{project_prefix}-reward-prompt",
    type=REWARD_PROMPT,
    source=reward_prompt_uri,
    wait=True
)

print(f"Reward prompt ARN: {reward_prompt.arn}")
```

This creates a versioned, addressable resource (identified by its ARN) that can be referenced by the RLAIF training job.

### Reward Model Selection

We use **GPT OSS 120B** via Amazon Bedrock as our AI judge:

```python
reward_model_id = "openai.gpt-oss-120b-1:0"
```

This model is accessed through Bedrock during training to score each candidate response.

:::alert{header="Key Concept" type="info"}
The reward prompt is critical to RLAIF success. It defines what "good" means for your use case. Experiment with different prompts to optimize for your specific goals.
:::

---

Once the reward model is configured, you're ready to launch the RLAIF fine-tuning job.
