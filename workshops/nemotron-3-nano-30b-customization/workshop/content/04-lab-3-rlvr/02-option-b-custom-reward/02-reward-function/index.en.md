---
title: "Create the Reward Function"
weight: 2
---

## About custom reward functions

In RLVR, the **reward function** scores each model-generated response so the reinforcement learning algorithm (GRPO) can optimize toward higher-scoring outputs. Option A used the built-in `default_compute_score` (exact match). In this option you bring your own.

A SageMaker AI Registry **Reward Function** Evaluator can be created from either:

- an **AWS Lambda ARN**, or
- **bring-your-own code** — a local Python file that SageMaker packages and runs.

This lab uses bring-your-own code by passing the local `reward_function.py` path.

---

## Configure the reward function

::alert[📒 Open the notebook **`lab-3a-custom-reward-function-rlvr/2-create-reward-function.ipynb`**]

### The evaluator contract

A Reward Function evaluator is just a Python file with an entry point that SageMaker invokes on **batches** of model outputs during training. Two responsibilities define the contract:

1. **`lambda_handler(event, context)`** — the required entry point. It receives a batch of records and returns the results. This is where SageMaker calls in.
2. **The scoring logic** — for each record, turn one model response + its reference answer into a score. In this lab that logic lives in a helper called `reward_function(record, index)`, but the name is yours to choose — `lambda_handler` can call any function you write.

:::alert{header="Two-function pattern" type="info"}
Keep `lambda_handler` thin — parse the batch, loop, package the response — and put your task-specific scoring in a separate function. This keeps the SageMaker plumbing isolated from the logic you actually want to iterate on, and makes the scorer trivial to unit-test.
:::

#### Input schema — what SageMaker sends

`event` is a **list of records**. Each record carries the conversation (the **last `assistant` message is the model response to score**) plus any extra fields you included in your dataset — here, `reference_answer`:

```python
[
    {
        "messages": [
            {"role": "user", "content": "<the math problem>"},
            {"role": "assistant", "content": "<the model's generated answer>"}  # score this
        ],
        "reference_answer": "42"  # extra field carried through from your dataset
    },
    ...
]
```

#### Output schema — what you must return

`lambda_handler` returns an API Gateway-style response. The `body` is a JSON **list** with one result per input record; each result needs an `id`, an `aggregate_reward_score`, and a `metrics_list`:

```python
return {
    "statusCode": 200,
    "headers": {"Content-Type": "application/json"},
    "body": json.dumps(results),  # list of per-record results (below)
}
```

```python
# one entry in `results`
{
    "id": record_id,                    # echo the record's id
    "aggregate_reward_score": 0.9,      # float 0.0–1.0 — THIS is what drives training
    "metrics_list": [
        {"name": "numeric_correctness", "value": 1.0, "type": "Reward"},
        {"name": "reasoning_present",   "value": 1.0, "type": "Metric"},
        # ...
    ],
}
```

- **`aggregate_reward_score`** is the single number GRPO optimizes. Everything else is for observability.
- **`metrics_list`** items are typed `"Reward"` (a component that feeds the reward) or `"Metric"` (tracked for insight only). Both show up in MLflow so you can see _why_ the score moved.

### Walking through `reward_function.py`

The file is organized as **small extract-then-score helpers**, a **per-record `reward_function`**, and a **thin `lambda_handler`**. Here are the core parts (snippets are condensed for readability — open `reward_function.py` for the full version, including debug logging and extra payload-shape guards).

**1. `lambda_handler` — parse the batch, loop, package the response.** Note it never scores anything itself; it delegates to `reward_function` per record and wraps everything in try/except so one bad record can't fail the whole batch:

```python
def lambda_handler(event, context):
    try:
        records = _as_batch(event)                       # normalize to a list
        results = [reward_function(rec, index=i)         # score each record
                   for i, rec in enumerate(records)]
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(results),
        }
    except Exception as error:
        return {"statusCode": 500, "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": str(error)})}
```

**2. Pull the model response out of the conversation.** The response to score is the last `assistant` turn in `messages`. A helper tolerates the different shapes a payload can take (a `messages` list, a raw string, a `response`/`completion` field, etc.) so the scorer is robust:

```python
def _last_assistant_message(messages):
    response = ""
    for message in messages or []:
        if message.get("role") == "assistant":
            response = str(message.get("content", ""))
    return response  # last assistant message wins
```

**3. Pull the reference answer.** `reference_answer` is the field this lab adds in data prep, but the helper also falls back to `ground_truth`, `reward_model.ground_truth`, or `extra_info` — so the same file works whether the answer arrives at the top level or nested:

```python
def _extract_reference_answer(record):
    for key in ("reference_answer", "ground_truth", "answer"):
        if record.get(key) not in (None, ""):
            return record[key]
    # ...also checks record["reward_model"]["ground_truth"] and record["extra_info"]
```

**4. Extract the model's final number.** Prefer the GSM8K `#### <number>` convention; if it's missing, fall back to the last number in the text. Comparison is done with `Decimal` (after stripping commas) so `1,000` and `1000.0` compare equal:

```python
def _extract_final_answer(response):
    final = _FINAL_ANSWER_PATTERN.search(response)   # r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)"
    if final:
        return _to_decimal(final.group(1))
    numbers = list(_NUMBER_PATTERN.finditer(response))
    return _to_decimal(numbers[-1].group(0)) if numbers else None
```

**5. Compute per-record signals and combine them.** This is the heart of the reward — and the part you would change for your own task. Numeric correctness dominates; parseability, format, and reasoning add a small **shaping** signal so that early, mostly-wrong rollouts still produce a non-zero gradient instead of collapsing to all-zeros:

```python
def reward_function(record, index=0):
    response = _extract_model_response(record)
    reference = _extract_reference_answer(record)

    predicted = _extract_final_answer(response)
    expected  = _to_decimal(reference)

    numeric_correctness = 1.0 if (predicted is not None and predicted == expected) else 0.0
    answer_parseable    = 1.0 if predicted is not None else 0.0
    final_answer_format = _has_final_answer_format(response)   # uses "#### <n>"?
    reasoning_present   = _has_reasoning(response)             # ≥ 8 words before "####"?

    aggregate_reward_score = round(
        (0.6 * numeric_correctness)   # correctness is most of the reward
        + (0.2 * answer_parseable)    # + shaping signals below
        + (0.1 * final_answer_format)
        + (0.1 * reasoning_present),
        4,
    )

    return {
        "id": str(record.get("id") or index),
        "aggregate_reward_score": aggregate_reward_score,
        "metrics_list": [
            {"name": "numeric_correctness", "value": numeric_correctness, "type": "Reward"},
            {"name": "answer_parseable",    "value": answer_parseable,    "type": "Reward"},
            {"name": "reference_available", "value": 1.0 if expected is not None else 0.0, "type": "Metric"},
            {"name": "final_answer_format", "value": final_answer_format, "type": "Metric"},
            {"name": "reasoning_present",   "value": reasoning_present,   "type": "Metric"},
        ],
    }
```

The five signals it emits:

| Signal                | Type   | Weight | Meaning                                                       |
| --------------------- | ------ | ------ | ------------------------------------------------------------- |
| `numeric_correctness` | Reward | 0.6    | 1.0 if the model's final number equals the reference answer   |
| `answer_parseable`    | Reward | 0.2    | 1.0 if a final numeric answer could be extracted              |
| `final_answer_format` | Metric | 0.1    | 1.0 if the response uses the `#### <number>` convention       |
| `reasoning_present`   | Metric | 0.1    | 1.0 if there is non-trivial reasoning before the final answer |
| `reference_available` | Metric | —      | 1.0 if a reference answer was present (data-quality check)    |

### Adapting it to your own task

To write a reward function for a different task, keep `lambda_handler` and the input/output schema exactly as above, and rewrite only the scoring logic:

- **Change what "correct" means** — swap numeric matching for string/regex checks, a unit test pass/fail, an embedding similarity, an external API call, etc.
- **Reweight or add signals** — adjust the coefficients, or add new `metrics_list` entries (`"Reward"` to influence training, `"Metric"` to only observe).
- **Return a continuous score** — `aggregate_reward_score` need not be 0/1; partial credit (e.g. fraction of unit tests passed) often trains better.
- **Stay defensive** — models emit malformed output; guard your parsing and never let one record raise out of the batch.

### Smoke-test the reward function locally

Before registering, the notebook runs the function locally on a small batch to confirm a correct answer scores `1.0` and a wrong answer scores lower:

```python
import json
from reward_function import lambda_handler

sample_event = [
    {
        "id": "gsm8k-smoke-000001",
        "messages": [
            {"role": "user", "content": "What is 6 times 7? Think step by step and output the final answer after ####."},
            {"role": "assistant", "content": "6 times 7 is 42, so the final answer is #### 42"},
        ],
        "reference_answer": "42",
    },
    {
        "id": "gsm8k-smoke-000002",
        "messages": [
            {"role": "user", "content": "What is 10 plus 5?"},
            {"role": "assistant", "content": "The answer is #### 12"},
        ],
        "reference_answer": "15",
    },
]

response = lambda_handler(sample_event, None)
results = json.loads(response["body"])
assert response["statusCode"] == 200
assert results[0]["aggregate_reward_score"] == 1.0
assert results[1]["aggregate_reward_score"] < 1.0
print("Smoke test passed.")
```

### Register the reward function evaluator

The notebook registers the local `reward_function.py` as a `REWARD_FUNCTION` Evaluator in the SageMaker AI Registry:

```python
from sagemaker.ai_registry.air_constants import REWARD_FUNCTION
from sagemaker.ai_registry.evaluator import Evaluator

reward_function_evaluator = Evaluator.create(
    name=reward_function_name,
    type=REWARD_FUNCTION,
    source="reward_function.py",
    role=role,
    sagemaker_session=sess,
    wait=True,
)

reward_function_evaluator.refresh()
print(f"Reward function ARN: {reward_function_evaluator.arn}")
```

This creates a versioned, addressable resource (identified by its ARN) that the RLVR training job references in the next step.

:::alert{header="Keep the evaluator name short" type="warning"}
SageMaker creates an underlying Lambda function named `SageMaker-evaluator-<name>-<timestamp>`, and Lambda function names must be ≤ 64 characters. The notebook uses a short name (`gsm8k-rlvr-rf`) to stay within the limit.
:::

:::alert{header="Key Concept" type="info"}
The reward function defines what "good" means for your task. The aggregate score drives training, while the additional metrics give you observability into _why_ a response scored the way it did.
:::

---

Once the reward function is registered, you're ready to launch the RLVR fine-tuning job.
