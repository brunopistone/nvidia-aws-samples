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

![LLM-as-a-Judge](/static/images/lab-1-sft/llm_judge.png)

LLM-as-a-Judge uses a large language model (here **Amazon Nova Pro**) to evaluate outputs against human-aligned criteria that statistical metrics can't capture.

Traditional metrics measure word overlap but miss:

- Factual errors
- Hallucinations
- Task-specific quality dimensions

LLM-as-a-Judge addresses this by scoring multiple quality dimensions and returning an explanation for each score.

---

## Option 1: Evaluate with Code

::alert[📒 Open the notebook **`code/3-evaluation.ipynb`**]

::alert[Prerequisite: a successful run of the fine-tuning notebook. This lab retrieves the fine-tuned model from the Model Package Group and the test dataset from the AI Registry.]

:::alert{header="Required IAM setup for Bedrock" type="warning"}
The evaluation job runs LLM-as-a-Judge scoring on Amazon Bedrock, so your **SageMaker execution role must list `bedrock.amazonaws.com` as a trusted entity** in its trust policy.

Without it the pipeline appears to start normally but the `EvaluateCustomModelMetrics` step fails. You will only see the error by inspecting the pipeline execution in the SageMaker console (**Pipelines** → **Executions** → select the failed step).

See [Amazon Bedrock permissions setup](https://docs.aws.amazon.com/bedrock/latest/userguide/judge-service-roles.html) for how to update the trust relationship. If you are at an AWS-led event, this is already configured for you.
:::

### Retrieve the fine-tuned model

The notebook rebuilds the same hash-truncated Model Package Group name used during fine-tuning, then takes the most recent model package in it:

```python
response = sm_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)

fine_tuned_model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
fine_tuned_model_package_group_arn = ModelPackageGroup.get(
    model_package_group_name
).model_package_group_arn

test_dataset = DataSet.get(name="Multilingual-Thinking-sft-test")
```

### Define Custom Metrics

Built-in metrics cover general quality. The custom metrics below score the **specific behaviour** this model was fine-tuned for — reasoning inside `<think>` tags in a target non-English language, then answering in English:

```python
EVALUATOR_MODEL = "amazon.nova-pro-v1:0"
BUILTIN_METRICS = ["Correctness", "Completeness", "Faithfulness", "Coherence"]
```

| Custom metric                | What it checks                                                                      |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `ReasoningLanguageAdherence` | The reasoning inside `<think>` is in the same non-English language as the reference |
| `EnglishFinalAnswer`         | The final answer (after `</think>`) is in fluent, natural English                   |
| `ThinkTagStructure`          | Exactly one well-formed `<think>...</think>` block, followed by a non-empty answer  |
| `ReasoningQuality`           | The reasoning is on-topic, coherent, and logically supports the final answer        |

```python
custom_metrics_list = [
    {
        "customMetricDefinition": {
            "name": "ReasoningLanguageAdherence",
            "instructions": (
                "The model was asked to write its reasoning (the text inside the "
                "<think>...</think> tags) in a specific non-English language. "
                "The reference response shows that target language. "
                "First, identify the language used inside the <think>...</think> tags "
                "of the reference. Then check whether the model's response writes its "
                "reasoning inside its own <think>...</think> tags in that SAME language. "
                "Judge the language only, not the correctness of the content.\n\n"
                "Prompt: {{prompt}}\n"
                "Response: {{prediction}}\n"
                "Reference: {{ground_truth}}"
            ),
            "ratingScale": [
                {
                    "definition": "Reasoning is in the same non-English target language as the reference",
                    "value": {"floatValue": 1},
                },
                {
                    "definition": "Reasoning is in a different language, or there is none to assess",
                    "value": {"floatValue": 0},
                },
            ],
        }
    },
    {
        "customMetricDefinition": {
            "name": "EnglishFinalAnswer",
            "instructions": (
                "Consider ONLY the final answer, i.e. the text that appears AFTER the "
                "closing </think> tag in the response. The model was instructed to give "
                "this final answer in English, regardless of the reasoning language. "
                "Judge the language of the final answer only, not its correctness.\n\n"
                "Prompt: {{prompt}}\n"
                "Response: {{prediction}}"
            ),
            "ratingScale": [
                {
                    "definition": "The final answer after </think> is in fluent, natural English",
                    "value": {"floatValue": 1},
                },
                {
                    "definition": "The final answer is in another language, empty, or missing",
                    "value": {"floatValue": 0},
                },
            ],
        }
    },
    # ThinkTagStructure and ReasoningQuality follow the same pattern — see the notebook
]

custom_metrics_json = json.dumps(custom_metrics_list)
```

:::alert{header="Only three placeholders are available" type="info"}
Custom metric instructions can reference `{{prompt}}` (the input), `{{prediction}}` (the model's output) and `{{ground_truth}}` (the reference answer) — there is **no `{{system}}` placeholder**.

The target reasoning language lives in the dataset's `system` field, which the judge never sees. That's why `ReasoningLanguageAdherence` asks the judge to _infer_ the expected language from the reference completion's `<think>` block instead of being told it directly.
:::

### Run Evaluation

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
    evaluate_base_model=False,  # skip the base model to evaluate only the custom model
    sagemaker_session=sess,
)

execution = evaluator.evaluate()
```

`evaluate()` launches a serverless evaluation pipeline that scores the fine-tuned model against the test dataset with both metric sets. The job appears under the model's **Evaluation** tab in Studio, where you can track it to completion.

:::alert{header="Timing" type="info"}
The evaluation pipeline takes roughly **15-20 minutes**. Monitor progress in the SageMaker console under **Pipelines** → **Executions**.
:::

### Analyze Results

Keep a reference to the execution returned by `evaluate()` — `get_all()` returns executions from every LLM-as-a-Judge pipeline in the account with no time ordering, so filtering only by "Succeeded" can pick up an unrelated run:

```python
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

latest_succeeded.show_results(limit=5, offset=0, show_explanations=False)
```

### Download and Visualize Results

The notebook locates this run's results file under the evaluation output prefix and downloads it locally:

```python
# looks for keys ending in "_output.jsonl" containing "custom-llmaj-eval",
# preferring files whose key contains this execution's id
s3_client.download_file(bucket, jsonl_key, "./tmp/evaluation_results.jsonl")
```

Then it aggregates the per-record scores and renders three views:

```python
df = load_evaluation_results("./tmp/evaluation_results.jsonl")
plot_metrics_bar(df)
plot_metrics_radar(df)
plot_metrics_bullet(df, target=0.8)
```

These visualizations help you understand:

- Average scores across all metrics
- Performance relative to a target threshold
- Strengths and weaknesses of the fine-tuned model

---

## Option 2: Evaluate with UI

After training completes, you can evaluate the fine-tuned model directly from the Studio UI. From the training job detail page, choose **Go to Custom Model**, then **Evaluate** to launch the evaluation wizard.

The wizard lets you:

- Choose the evaluation type: **LLM-as-a-Judge**, **Custom Scorer**, or **Benchmarks**
- Select an evaluator model (Nova Pro, Claude, Mistral)
- Upload a test dataset or select an existing one
- Configure the output location

:::alert{header="Note" type="info"}
In this workshop we use the SDK approach, because the task-specific custom metrics above are easier to define and version in code.
:::
