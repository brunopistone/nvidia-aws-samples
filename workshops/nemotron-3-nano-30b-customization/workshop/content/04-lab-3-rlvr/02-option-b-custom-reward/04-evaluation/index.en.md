---
title: "Evaluation"
weight: 4
---

After RLVR training, evaluation tells you whether your custom reward function actually improved the model's mathematical reasoning.

## Why benchmark evaluation?

RLVR trains models on tasks with objectively verifiable answers, so it makes sense to evaluate them the same way — using a standardized benchmark with known correct answers rather than subjective LLM-as-a-Judge scoring. We use the **MATH benchmark** to measure the fine-tuned model's mathematical reasoning.

---

## Evaluate with Code

::alert[📒 Open the notebook **`lab-3a-custom-reward-function-rlvr/4-evaluation.ipynb`**]

### Retrieve the custom-reward fine-tuned model

The notebook locates the latest model package in the model package group created by the training notebook. To evaluate a specific version, set `fine_tuned_model_package_arn` manually before the lookup cell:

```python
base_model_id = "huggingface-reasoning-qwen3-06b"
model_package_group_name = f"{base_model_id}-custom-rw-rlvr"
fine_tuned_model_package_arn = None

if fine_tuned_model_package_arn is None:
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    fine_tuned_model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
```

### Explore Available Benchmarks

List the available benchmarks and inspect the MATH benchmark properties:

```python
from sagemaker.train.evaluate import BenchMarkEvaluator, get_benchmarks, get_benchmark_properties
from rich.pretty import pprint

Benchmark = get_benchmarks()
pprint(list(Benchmark))

# View MATH benchmark properties
pprint(get_benchmark_properties(benchmark=Benchmark.MATH))
```

### Create Benchmark Evaluator

Use the `BenchMarkEvaluator` to evaluate the fine-tuned model against the MATH benchmark:

```python
evaluator = BenchMarkEvaluator(
    benchmark=Benchmark.MATH,
    model=fine_tuned_model_package_arn,
    model_package_group=model_package_group_name,
    base_eval_name="custom-reward-rlvr",
    s3_output_path=output_path,
    evaluate_base_model=False,
    role=role,
    sagemaker_session=sess,
)
```

### Run Evaluation

```python
execution = evaluator.evaluate()
execution.wait()
```

:::alert{header="Important" type="warning"}
The benchmark evaluation can take **15-30 minutes** to complete.
:::

### View Results

This option runs a single benchmark evaluation, so we display the one we just launched. Prefer the `execution` object returned by `evaluator.evaluate()`; if it isn't in scope, fall back to looking it up by its S3 output path (`get_all()` returns executions from every benchmark pipeline in the account and isn't ordered by time, so a plain `[-1]` could pick an unrelated run):

```python
from sagemaker.train.evaluate import EvaluationPipelineExecution
from sagemaker.train.evaluate.constants import EvalType

try:
    result_execution = execution
except NameError:
    result_execution = next(
        (
            e
            for e in EvaluationPipelineExecution.get_all(eval_type=EvalType.BENCHMARK)
            if e.status.overall_status == "Succeeded"
            and getattr(e, "s3_output_path", None) == output_path
        ),
        None,
    )

pprint(result_execution)
pprint(result_execution.show_results())
```

The benchmark evaluator outputs a `results_*.json` file containing `math_exact_match` scores for each MATH category, so you can see how the custom-reward fine-tuned model performs across areas of mathematical reasoning.

:::alert{header="Note" type="info"}
Because this option uses a small base model (Qwen 3 0.6B) and a small training set for fast workshop runs, the goal is to demonstrate the custom-reward workflow end-to-end rather than to maximize benchmark scores.
:::

:::alert{header="Short on time?" type="warning"}
If you don't want to wait for the benchmark evaluation job to complete, the notebook supports downloading pre-computed evaluation results so you can skip ahead to viewing the scores.
:::

---

Once evaluation completes, you're ready to deploy the fine-tuned model.
