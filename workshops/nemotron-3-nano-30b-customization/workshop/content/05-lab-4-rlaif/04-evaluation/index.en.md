---
title: "Model Evaluation"
weight: 4
---

## Evaluate with LLM-as-a-Judge

::alert[📒 Open the notebook **`lab-4-reinforcement-learning-from-ai-feedback/4-evaluation.ipynb`**]

### Retrieve the Fine-Tuned Model Package

The notebook rebuilds the Model Package Group name (the same hashing/truncation logic used during fine-tuning), then fetches the most recent Model Package in the group as the model to evaluate:

```python
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.core.resources import ModelPackageGroup

response = sm_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)

if len(response["ModelPackageSummaryList"]) > 0:
    fine_tuned_model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
    fine_tuned_model_package_group_arn = ModelPackageGroup.get(model_package_group_name).model_package_group_arn
else:
    fine_tuned_model_package_arn = None
    fine_tuned_model_package_group_arn = None

if default_prefix:
    s3_output_path = f"s3://{bucket_name}/{default_prefix}/{project_prefix}-{base_model_shortname}/eval"
else:
    s3_output_path = f"s3://{bucket_name}/{project_prefix}-{base_model_shortname}/eval"
```

### Define Custom Evaluation Metrics

We use LLM-as-a-Judge to evaluate how human-like the model responses are:

```python
custom_metrics_list = [
    {
        "customMetricDefinition": {
            "name": "human-like-alignment",
            "instructions": (
                "You are an expert at evaluating language model outputs and "
                "determining which responses sound more natural and human-like. "
                "Be extremely critical and look at it very thoroughly.\n\n"
                "You may use the ground truth response as a reference of what "
                "a human-like answer should contain.\n\n"
                "Focus on these aspects when evaluating:\n"
                "- Human-like reasoning and explanations\n"
                "- Emotional intelligence and empathy where relevant\n"
                "- Avoidance of overly rigid or mechanical language\n"
                "- Natural conversation flow\n"
                "- Appropriate level of formality\n\n"
                "Here is the actual task:\n"
                "Task: {{prompt}}\n"
                "Ground Truth Response: {{ground_truth}}\n"
                "Candidate Response: {{prediction}}"
            ),
            "ratingScale": [
                {
                    "definition": "No part of the response sounds like a human-like response.",
                    "value": {"floatValue": 0}
                },
                {
                    "definition": "Roughly half of the response is a human-like response.",
                    "value": {"floatValue": 1}
                },
                {
                    "definition": "Every piece of the response sounds like a human-like response.",
                    "value": {"floatValue": 2}
                }
            ]
        }
    }
]
```

Note: The evaluation dataset uses `query` and `response` fields. The SDK maps these to the template variables `{{prompt}}`, `{{ground_truth}}`, and `{{prediction}}` automatically.

### Run Evaluation

Evaluate both the base model and fine-tuned model with a single evaluator:

```python
from sagemaker.train.evaluate import LLMAsJudgeEvaluator
from sagemaker.ai_registry.dataset import DataSet

evaluator_model_id = "amazon.nova-pro-v1:0"
eval_dataset = DataSet.get(f"{project_prefix}-eval")

custom_metrics_json = json.dumps(custom_metrics_list)

evaluator = LLMAsJudgeEvaluator(
    model=fine_tuned_model_package_arn,
    model_package_group=fine_tuned_model_package_group_arn,
    evaluator_model=evaluator_model_id,
    dataset=eval_dataset.arn,
    custom_metrics=custom_metrics_json,
    sagemaker_session=sess,
    s3_output_path=s3_output_path,
    evaluate_base_model=True  # Automatically triggers a second evaluation using the base model
)

execution = evaluator.evaluate()
```

### Download and Analyze Results

Download evaluation results from S3:

```python
import os

execution_id = execution.arn.split("/")[-1]
output_path = execution.s3_output_path

os.makedirs("./eval_results", exist_ok=True)

s3 = boto3.client("s3", region_name=boto3.Session().region_name)

bucket = output_path.replace("s3://", "").split("/")[0]
prefix = "/".join(output_path.replace("s3://", "").split("/")[1:])

base_key = None
custom_key = None

for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix)["Contents"]:
    key = obj["Key"]
    if execution_id in key and "_output.jsonl" in key and "/datasets/" in key:
        if "base-llmaj-eval" in key:
            base_key = key
        elif "custom-llmaj-eval" in key:
            custom_key = key

s3.download_file(bucket, base_key, "./eval_results/base_eval.jsonl")
s3.download_file(bucket, custom_key, "./eval_results/custom_eval.jsonl")
```

### Compare Results

Analyze the improvement from RLAIF fine-tuning:

```python
import json

def get_score(result):
    scores = result.get('automatedEvaluationResult', {}).get('scores', [])
    for score in scores:
        if score.get('metricName') == 'human-like-alignment':
            return score.get('result')
    return None

with open('./eval_results/base_eval.jsonl') as f:
    base_results = [json.loads(line) for line in f]

with open('./eval_results/custom_eval.jsonl') as f:
    custom_results = [json.loads(line) for line in f]

base_scores = [get_score(r) for r in base_results if get_score(r) is not None]
custom_scores = [get_score(r) for r in custom_results if get_score(r) is not None]

# Calculate average scores (0.0, 1.0, 2.0 scale)
base_avg = sum(base_scores) / len(base_scores) / 2 * 100
custom_avg = sum(custom_scores) / len(custom_scores) / 2 * 100

print(f"Base Model Average: {base_avg:.1f}%")
print(f"Fine-tuned Model Average: {custom_avg:.1f}%")
print(f"Improvement: {custom_avg - base_avg:+.1f}%")
```

### Visualize Results

Create comparison charts:

```python
import matplotlib.pyplot as plt
import numpy as np

# Count distribution
base_counts = [base_scores.count(0), base_scores.count(1), base_scores.count(2)]
custom_counts = [custom_scores.count(0), custom_scores.count(1), custom_scores.count(2)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Distribution comparison
categories = ['No part\n(0.0)', 'Roughly half\n(1.0)', 'All of it\n(2.0)']
x = np.arange(len(categories))
width = 0.35

ax1.bar(x - width/2, base_counts, width, label='Base Model')
ax1.bar(x + width/2, custom_counts, width, label='Fine-tuned Model')
ax1.set_xlabel('Human-like Rating')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Human-like Ratings')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()

# Average score comparison
ax2.bar(['Base Model', 'Fine-tuned Model'], [base_avg, custom_avg])
ax2.set_ylabel('Average Score (%)')
ax2.set_title('Average Human-like Alignment Score')
ax2.set_ylim([0, 100])

plt.tight_layout()
plt.show()
```

:::alert{header="Expected Results" type="info"}
RLAIF fine-tuning should show improvement in human-like alignment scores, with more responses rated as "All of it" (2.0) and fewer rated as "No part" (0.0).
:::

:::alert{header="Short on time?" type="warning"}
If you don't want to wait for the evaluation jobs to complete, the notebook supports downloading precomputed evaluation results so you can skip ahead to analyzing and visualizing the comparison.
:::

---

Once evaluation is complete, you're ready to deploy the fine-tuned model.
