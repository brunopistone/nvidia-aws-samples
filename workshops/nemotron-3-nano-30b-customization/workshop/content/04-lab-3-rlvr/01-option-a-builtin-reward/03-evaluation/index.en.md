---
title: "Evaluation"
weight: 3
---

After RLVR training, evaluation is critical to understand whether the model's mathematical reasoning has improved.

## Why Evaluation Matters

Without rigorous evaluation, you can't answer critical questions:

- Did RLVR training improve the model's ability to solve math problems?
- How much better is the fine-tuned model compared to the base model?
- Is the model ready for production deployment?

## Benchmark Evaluation

For RLVR-trained models, we use the **MATH benchmark** — a standardized evaluation that tests mathematical reasoning capabilities. Unlike LLM-as-a-Judge (used in the SFT lab), benchmark evaluation uses objective scoring against known correct answers, which aligns naturally with how RLVR trains models.

---

## Option 1: Evaluate with Code

::alert[📒 Open the notebook **`lab-3-reinforcement-learning-from-verifiable-rewards/3-evaluation.ipynb`**]

### Prerequisites

Retrieve the latest model package from the Model Package Group created during fine-tuning:

```python
from sagemaker.core.resources import ModelPackageGroup

base_model_id = "huggingface-reasoning-qwen3-06b"
model_package_group_name = f"{base_model_id}-rlvr"

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

# Separate output paths for the fine-tuned and base-model evaluation jobs
prefix = f"{default_prefix}/" if default_prefix else ""
output_path = f"s3://{bucket_name}/{prefix}{base_model_id}/evaluation"
output_path_base = f"s3://{bucket_name}/{prefix}{base_model_id}/benchmark-evaluation-base"
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

### Create and run the benchmark evaluator

We run the `BenchMarkEvaluator` **twice** — once for the fine-tuned model (a model-package ARN) and once for the base model (passed directly as its JumpStart model ID) — so we can compare the two. Each call sets `evaluate_base_model=False` because we evaluate the base model explicitly as its own job rather than letting a single evaluator do both:

```python
# Evaluate the fine-tuned model
evaluator = BenchMarkEvaluator(
    benchmark=Benchmark.MATH,
    model=fine_tuned_model_package_arn,
    model_package_group=model_package_group_name,
    base_eval_name="fine-tuned-model-rlvr",
    s3_output_path=output_path,
    evaluate_base_model=False,
    sagemaker_session=sess,
)
execution = evaluator.evaluate()

# Evaluate the base model using the JumpStart model ID directly
base_evaluator = BenchMarkEvaluator(
    benchmark=Benchmark.MATH,
    model=base_model_id,
    base_eval_name="base-model-rlvr",
    s3_output_path=output_path_base,
    evaluate_base_model=False,
    sagemaker_session=sess,
)
execution = base_evaluator.evaluate()
execution.wait()
```

:::alert{header="Important" type="warning"}
The benchmark evaluation can take **15-30 minutes** to complete when evaluating both models.
:::

### View Results

Retrieve the succeeded benchmark executions — the fine-tuned and base-model jobs — and display their results side by side:

```python
from rich.pretty import pprint
from sagemaker.train.evaluate import EvaluationPipelineExecution
from sagemaker.train.evaluate.constants import EvalType

# Get all succeeded evaluations and take the first 2
all_succeeded = [
    e for e in EvaluationPipelineExecution.get_all(eval_type=EvalType.BENCHMARK)
    if e.status.overall_status == "Succeeded"
]

last_two_succeeded = all_succeeded[:2]

for i, execution in enumerate(last_two_succeeded, 1):
    print(f"=== Succeeded Evaluation #{i} ===")
    pprint(execution)
    pprint(execution.show_results())
```

The results show the MATH benchmark scores (`math_exact_match` per category, such as algebra, geometry, and number theory), which you can use to see how RLVR training improved the model's mathematical reasoning compared to the base model.

## Option 2: Evaluate with UI

Once your training job is complete, you can launch evaluation directly from the Studio UI.

From the training job details, click **Go to Custom Model**:

![Go to Custom Model](/static/images/lab-3-rlvr/studio-job-go-to-model.png)

In the custom model view, click **Evaluate** to launch the evaluation wizard:

![Custom Model Evaluate](/static/images/lab-3-rlvr/studio-custom-model-evaluate.png)

The evaluation setup allows you to:

- Choose evaluation type: **LLM-as-a-Judge**, **Custom Scorer**, or **Benchmarks**
- Select a benchmark (e.g., MATH)
- Enable base model comparison
- Configure output location

![Evaluation Setup](/static/images/lab-3-rlvr/studio-evaluation-setup.png)

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach with the MATH benchmark to objectively measure mathematical reasoning improvements from RLVR training.
:::

:::alert{header="Short on time?" type="warning"}
If you don't want to wait for the benchmark evaluation jobs to complete, the notebook supports downloading pre-computed evaluation results so you can skip ahead to comparing the base and fine-tuned models.
:::
