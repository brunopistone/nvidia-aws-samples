---
title: "Evaluation"
weight: 3
---

After fine-tuning, evaluation is critical to understand whether the model has improved and by how much.

## Why Evaluation Matters

Without rigorous evaluation, you can't answer critical questions:

- Did fine-tuning improve performance on your target task?
- How much better is the fine-tuned model compared to the base model?
- Is the model ready for production deployment?

## LLM-as-a-Judge Evaluation

![LLM-as-a-Judge](/static/images/lab-2-dpo/llm_judge.png)

LLM-as-a-Judge uses a large language model (Amazon Nova Pro) to evaluate outputs based on human-aligned criteria that statistical metrics can't capture.

### Why LLM-as-a-Judge?

Traditional metrics measure word overlap but miss:

- Factual errors
- Hallucinations
- Task-specific quality dimensions

LLM-as-a-Judge addresses this by evaluating multiple quality dimensions with explanations.

---

## Alternative: Evaluate with UI

Once your training job is complete, you can launch evaluation directly from the Studio UI.

From the training job details, click **Go to Custom Model**:

![Go to Custom Model](/static/images/lab-2-dpo/studio-job-go-to-model.png)

In the custom model view, click **Evaluate** to launch the evaluation wizard:

![Custom Model Evaluate](/static/images/lab-2-dpo/studio-custom-model-evaluate.png)

The evaluation setup allows you to:

- Choose evaluation type: **LLM-as-a-Judge**, **Custom Scorer**, or **Benchmarks**
- Optionally choose to compare the fine-tuned model with the base model in the evaluation
- Select an evaluator model (Nova Pro, Claude, Mistral)
- Upload or select an existing test dataset
- Configure output location

![Evaluation Setup](/static/images/lab-2-dpo/studio-evaluation-setup.png)

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach to define custom metrics for human-likeness evaluation.
:::

---

## Evaluate with Code

::alert[📒 Open the notebook **`lab-2-direct-preference-optimization-DPO/3-dpo-evaluation.ipynb`**]

## Prerequisites

Retrieve the fine-tuned model and test dataset:

```python
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.core.resources import ModelPackageGroup

base_model_id = "meta-textgeneration-llama-3-2-1b-instruct"
model_package_group_name = f"{base_model_id}-dpo"

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

test_dataset = DataSet.get(name="humanlike-dpo-test")
```

## Define Custom Metrics

Create custom evaluation metrics for human-likeness:

```python
import json

EVALUATOR_MODEL = "amazon.nova-pro-v1:0"
BUILTIN_METRICS = ["Helpfulness", "Relevance", "Coherence"]

custom_metrics_list = [
    {
        "customMetricDefinition": {
            "name": "HumanLikeTone",
            "instructions": (
                "Evaluate if the response sounds like a friendly human conversation rather than "
                "a formal AI assistant. Human-like responses use casual language, show personality, "
                "and feel warm and approachable. Robotic responses use formal phrases like "
                "'I'm designed to', 'as an AI', 'I'm pleased to report', or overly corporate language. "
                "Prompt: {{prompt}}\nResponse: {{prediction}}"
            ),
            "ratingScale": [
                {"definition": "Excellent - Sounds completely natural and human", "value": {"floatValue": 3}},
                {"definition": "Good - Mostly human-like with minor formal elements", "value": {"floatValue": 2}},
                {"definition": "Mixed - Contains both human-like and robotic elements", "value": {"floatValue": 1}},
                {"definition": "Poor - Sounds robotic or like a corporate AI", "value": {"floatValue": 0}},
            ],
        }
    },
    {
        "customMetricDefinition": {
            "name": "ConversationalEngagement",
            "instructions": (
                "Assess if the response engages the user in natural conversation. "
                "Good responses ask follow-up questions, show genuine interest, and invite dialogue. "
                "Poor responses are one-sided or end abruptly without engagement. "
                "Prompt: {{prompt}}\nResponse: {{prediction}}"
            ),
            "ratingScale": [
                {"definition": "Highly engaging - Asks questions, invites dialogue", "value": {"floatValue": 2}},
                {"definition": "Somewhat engaging - Some conversational elements", "value": {"floatValue": 1}},
                {"definition": "Not engaging - One-sided, no conversational flow", "value": {"floatValue": 0}},
            ],
        }
    },
    {
        "customMetricDefinition": {
            "name": "AvoidRoboticPatterns",
            "instructions": (
                "Check if the response avoids robotic AI patterns. "
                "PENALIZE responses containing: 'As an AI/language model', "
                "'I'm designed to', 'I'm pleased to report', "
                "'I don't have personal experiences/emotions', or overly formal corporate speak. "
                "Prompt: {{prompt}}\nResponse: {{prediction}}"
            ),
            "ratingScale": [
                {"definition": "Good - No robotic patterns, sounds naturally human", "value": {"floatValue": 1}},
                {"definition": "Bad - Contains robotic AI patterns or formal self-references", "value": {"floatValue": 0}},
            ],
        }
    },
]

custom_metrics_json = json.dumps(custom_metrics_list)
```

## Run Evaluation

Use the `LLMAsJudgeEvaluator` to run the evaluation:

```python
from sagemaker.train.evaluate import LLMAsJudgeEvaluator

evaluator = LLMAsJudgeEvaluator(
    model=fine_tuned_model_package_arn,
    model_package_group=fine_tuned_model_package_group_arn,
    evaluator_model=EVALUATOR_MODEL,
    dataset=test_dataset,
    builtin_metrics=BUILTIN_METRICS,
    custom_metrics=custom_metrics_json,
    s3_output_path=output_path,
    evaluate_base_model=False,
    sagemaker_session=sess,
)

execution = evaluator.evaluate()
```

## Analyze Results

Retrieve the execution we just launched rather than an arbitrary one. `get_all()` returns executions from every LLM-as-Judge pipeline in the account with no time ordering, so prefer the `execution` object returned by `evaluator.evaluate()`; fall back to matching this job's `s3_output_path`:

```python
from sagemaker.train.evaluate import EvaluationPipelineExecution
from sagemaker.train.evaluate.constants import EvalType

try:
    latest_succeeded = execution
except NameError:
    latest_succeeded = next(
        (
            e
            for e in EvaluationPipelineExecution.get_all(eval_type=EvalType.LLM_AS_JUDGE)
            if e.status.overall_status == "Succeeded"
            and getattr(e, "s3_output_path", None) == output_path
        ),
        None,
    )
pprint(latest_succeeded)

latest_succeeded.show_results(limit=5, offset=0, show_explanations=False)
```

## Visualize Results

Download the results file this run produced, then visualize the metrics. The results live under a folder named with the **pipeline execution id** (the tail of the execution ARN), so we scope the search by that id and only match files under a `custom-llmaj-eval` folder (to avoid picking up the CustomInference step's own `inference_output.jsonl`):

```python
import os
import boto3
from urllib.parse import urlparse

s3_client = boto3.client("s3")
parsed = urlparse(latest_succeeded.s3_output_path)
bucket = parsed.netloc
prefix = parsed.path.lstrip("/")

execution_id = latest_succeeded.arn.split("/")[-1]

paginator = s3_client.get_paginator("list_objects_v2")
candidates = []
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith("_output.jsonl") and "custom-llmaj-eval" in key:
            priority = 1 if execution_id in key else 0  # prefer this run, keep others as fallback
            candidates.append((priority, obj["LastModified"], key))

jsonl_key = max(candidates)[2]  # highest priority (this run), then most recent
os.makedirs("./tmp", exist_ok=True)
s3_client.download_file(bucket, jsonl_key, "./tmp/evaluation_results.jsonl")
```

The notebook then loads the file and plots the metrics:

```python
df = load_evaluation_results("./tmp/evaluation_results.jsonl")
plot_metrics_bar(df)
plot_metrics_radar(df)
plot_metrics_bullet(df, target=0.8)
```

These visualizations help you understand:

- Average scores across all metrics
- Performance relative to target thresholds
- Strengths and weaknesses of the fine-tuned model

You an also visualize the results directly in the Studio UI

![Evaluation result](/static/images/lab-2-dpo/studio-eval-results.png)
![Evaluation result prompt](/static/images/lab-2-dpo/studio-eval-one-prompt.png)

:::alert{header="Short on time?" type="warning"}
If you don't want to wait for the evaluation job to complete, the notebook supports downloading pre-computed evaluation results so you can skip ahead to visualizing the metrics.
:::
