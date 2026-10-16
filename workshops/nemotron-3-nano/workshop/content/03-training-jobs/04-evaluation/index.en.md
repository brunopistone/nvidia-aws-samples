---
title: "Evaluation"
weight: 4
---

## What you will learn

You will evaluate the fine-tuned 4B model on held-out contracts, compare its verdict and citation scores with shipped reference runs, and optionally submit generated answers to an Amazon Bedrock LLM-as-a-Judge evaluation. You will also distinguish parsing from schema compliance, understand per-contract metric aggregation, inspect missing judgments, and preserve the evidence needed to interpret your own run.

::alert[Open [02-smtj-workshop/4-evaluation.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/4-evaluation.ipynb). The current source order is `1-prepare-data.ipynb`, `2-fine-tune-llm.ipynb`, `3-deployment.ipynb`, then `4-evaluation.ipynb`. Endpoint evaluation is implemented without AI Registry datasets or Model Packages. Choose shipped reference analysis or explicitly enable live evaluation as described below; the presence of the notebook is not evidence that your endpoint has been evaluated.]{type="info"}

The notebook imports [evaluation_utils.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/evaluation_utils.py) as `E`. That module owns endpoint requests, ordered prediction collection, gold-reference construction, score summaries, baseline loading, judge rubrics/input uploads/result readers, charts, and error diagnostics. The notebook retains execution switches, the actual `create_evaluation_job` submission, streaming/cost controls, and cleanup. Importing the utility creates no AWS clients or jobs; calling its inference or upload functions does. Keep this track's notebook and helper together, since the serverless file with the same name has a different API.

| Implementation                                             | Responsibility                                                                     |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `E.ask_endpoint`, `E.score_all`, `E.summarise`             | Generate raw text, score each contract, and average the four fields                |
| `E.load_rows`, `E.load_precomputed`                        | Read shipped JSON files without regenerating or rescoring answers                  |
| `E.build_judge_metrics`, `E.build_judge_dataset`           | Construct rubric definitions and upload precomputed-response JSONL                 |
| `E.wait_for_judge`, `E.read_judge_scores`, `E.judge_table` | Monitor jobs, read returned scores, and display means with coverage                |
| `contractnli_scorer.py`                                    | Tolerant parsing and deterministic per-contract equations, still separate from `E` |
| `inference_performance.py`                                 | Streaming measurements and rate-based compute allocation, not quality scoring      |

## 1. Choose what to run

The notebook offers two independent quality-evaluation switches, plus a separate opt-in streaming performance switch. Statistical evaluation invokes your existing SageMaker endpoint when enabled, while the judge evaluates answers that have already been generated. With both switches disabled, the analysis reads the files supplied in [baselines/](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/tree/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/baselines), not predictions from your deployment.

| `RUN_ENDPOINT_EVAL` | `RUN_JUDGE_EVAL` | What you evaluate                                                                                                              |
| ------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `False`             | `False`          | Shipped tuned predictions, base/frontier statistical summaries, and shipped base/tuned judge scores                            |
| `True`              | `False`          | Fresh tuned endpoint predictions; the judge section still displays historical scores, not judgments of those fresh predictions |
| `False`             | `True`           | Two new Bedrock jobs judging shipped base and tuned predictions; no endpoint invocation is needed                              |
| `True`              | `True`           | Fresh tuned endpoint predictions and two new judge jobs comparing shipped base predictions with your fresh tuned predictions   |

The switches do not regenerate the untuned or frontier baseline. The baseline comparison always reads their stored statistical summaries, and the live judge always uses stored untuned answers. There is no live frontier-model invocation or managed SageMaker evaluation pipeline in this notebook.

::alert[Do not run through the final **Clean up** cell until you intend to delete hosting resources. It is not guarded by either switch and attempts deletion even when both are `False`. Those defaults prevent endpoint inference and Bedrock evaluation-job submission, not all AWS activity: setup still resolves an execution role and default S3 bucket, and dataset setup can download files.]{type="warning"}

### Execution order

1. Use the Python 3 notebook environment prepared for this track, with the working directory set to `workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop`. The notebook uses relative paths for its helpers, raw dataset, and baselines.
2. Use this directory's [requirements.txt](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/requirements.txt), not the training-container requirements. Notebook imports include SageMaker SDK v3, Boto3/Botocore, `tqdm`, NumPy, and Matplotlib; the requirements specify `sagemaker==3.21.0` and load the parent `constraints-notebook.txt`. Verify imports in the actual prepared kernel; constraints restrict installed versions but do not independently install every listed package. The evaluation notebook has no dependency-installation cell.
3. For fresh endpoint scoring, complete [Deployment](/03-training-jobs/03-deployment/) using the actual [3-deployment.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/3-deployment.ipynb). Confirm the intended endpoint is `InService` and its one-contract smoke test succeeds. Leave it running until predictions have been collected.
4. Run evaluation Setup, dataset loading, and `E.check_scorer` before statistical scoring. Inspect the one-contract request before `E.score_all`. Run the optional performance/cost cells only if wanted, then the baseline comparison, judge definitions and selected judge branch, judge analysis, and error inspection. Chart colors are module constants; later plots no longer depend on a prior chart cell defining them.
5. Save the results and identities you need before deliberately running cleanup. Reading only reference results does not require deploying a model, but the notebook's setup still expects an AWS-configured environment.

### Permissions and resource boundaries

Use the same account and Region as the deployment. `Session()` determines the Region and `sess.default_bucket()` resolves the bucket. The setup obtains `get_execution_role()` and, on `ValueError`, looks up the IAM role named `sagemaker_execution_role`. That fallback requires the role to exist and the caller to be allowed to read it; it does not create or configure a role.

| Operation                                   | Access to verify before enabling it                                                                                                                                                                                       |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Notebook setup                              | Credentials, Region, execution-role resolution, and access needed to resolve or create the SageMaker default bucket                                                                                                       |
| Endpoint scoring                            | `sagemaker:InvokeEndpoint` on the intended endpoint; describe access for inspecting endpoint, configuration, and Model lineage                                                                                            |
| Judge submission and monitoring             | `bedrock:CreateEvaluationJob`, `bedrock:GetEvaluationJob`, and `iam:PassRole` for the approved evaluation role; list/stop permissions when inspecting or stopping jobs                                                    |
| Judge service role                          | Trust allowing `bedrock.amazonaws.com` to assume the role, invocation permission for the configured evaluator inference profile and its applicable backing models, and scoped access to read input and write output in S3 |
| Judge input/output handling by the notebook | `s3:PutObject` for input keys, `s3:ListBucket` for the output prefix, and `s3:GetObject` for results; applicable KMS permissions if encryption requires them                                                              |
| Hosting cleanup                             | Describe/delete access for the specific SageMaker Endpoint, EndpointConfig, and Model                                                                                                                                     |

The notebook passes the same `role` to Bedrock that setup resolved for SageMaker. SageMaker trust alone is insufficient, and a trust policy alone does not grant data or model access. Have the account administrator verify the caller and service-role permissions separately. Check evaluator access and evaluation-job quotas in the selected Region before submitting billed work. No registry registration, Model Package creation, scorer Lambda deployment, or new endpoint provisioning occurs in evaluation.

## 2. Verify model identity and the held-out inputs

### Endpoint identity is not model lineage

[config.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/config.py) sets `BASE_MODEL_ID` to `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and derives `MODEL_SLUG` by removing the organization prefix and replacing non-alphanumeric characters with hyphens. Evaluation calls `E.rname`, which matches deployment's naming algorithm: a 63-character limit and a six-character SHA-1 digest when truncation is necessary.

These assignments come from evaluation Setup:

```python
import evaluation_utils as E
from config import MODEL_SLUG

ENDPOINT_NAME = E.rname(f"{MODEL_SLUG}-contractnli", "-sft-ep")
SERVED_MODEL_NAME = "nemotron-contractnli"   # SM_VLLM_SERVED_MODEL_NAME in notebook 3
```

`ENDPOINT_NAME` selects the SageMaker resource. `SERVED_MODEL_NAME` is the separate OpenAI-compatible `model` field that vLLM validates against `SM_VLLM_SERVED_MODEL_NAME`. Neither is a Hugging Face model ID, training-job name, or Model Package ARN. There is no `InferenceComponentName` because the endpoint's production variant owns the Model directly.

Deployment resolves a completed training job through `TRAIN_JOB_PREFIX` or an explicit `TRAINING_JOB_NAME`, obtains `ModelArtifacts.S3ModelArtifacts`, and serves that merged `model.tar.gz`. Evaluation does not repeat this lookup or read a deployment-state file; it reconstructs a stable endpoint name in a fresh kernel. Record the training-job name, artifact URI, endpoint/configuration/Model names, container settings, account, and Region yourself. Deployment's get-first Model reuse does not verify that an existing Model points to a newly selected artifact, so a matching endpoint name alone is not proof that you are evaluating your latest run.

### Load raw ContractNLI, not the staged conversational JSONL

Run the notebook's dataset cell:

```python
C.ensure_dataset("./data")
EVAL_DOCS, labels = C.load("test")
EVAL_DOCS = EVAL_DOCS[:RECORDS]
label_keys = list(labels)
print(f"{len(EVAL_DOCS)} contracts, {len(label_keys)} checklist items each")
E.check_scorer(EVAL_DOCS[0], label_keys)
```

With the unmodified dataset and `RECORDS = 123`, expect the shape check to print `123 contracts, 17 checklist items each`, or 2,091 verdict decisions. `RECORDS` selects the first records in dataset order, not a random sample. Keep the test split separate from training and hyperparameter selection.

[contractnli.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/contractnli.py) reads `./data/contract-nli/test.json` by default, returning its `documents` and `labels`. Each document contains `text`, `spans` character-offset pairs, and `annotation_sets`; `C.gold_for(doc)` selects the first annotation set's `annotations`. `C.doc_spans(doc)` renders numbered text spans using their original zero-based indices. The environment variable `CONTRACTNLI_DIR`, if set before import, overrides the read directory, but `C.ensure_dataset("./data")` still downloads to `./data`. Verify those locations agree rather than assuming the override redirects the download.

[Data preparation](/03-training-jobs/01-data-preparation/) also writes `./tmp/test.jsonl`, with `prompt` system/user turns, an assistant `completion`, and `chat_template_kwargs: {"enable_thinking": false}`. The current evaluation notebook does **not** open that JSONL. It rebuilds the prompts and references from raw test documents using the same helpers. Preserve the helper and dataset versions used for training; editing them between notebooks can change the evaluation despite the shared function names. In the conversational file, `completion[0]["content"]` is the gold answer, never inference input.

The following is a shortened, synthetic answer-schema fixture used in the offline scoring check below, not a model result or a complete 17-item answer:

```json
{
  "nda-5": { "label": "Contradiction", "evidence": [3, 4] },
  "nda-4": { "label": "Entailment", "evidence": [5] },
  "nda-1": { "label": "NotMentioned", "evidence": [] }
}
```

The complete answer must contain exactly the keys in `labels`. Each entry should have one of `Entailment`, `Contradiction`, or `NotMentioned` and a list of integer span IDs from that contract. The prompt asks for an empty evidence list for `NotMentioned` and JSON only. Counting 17 entries is not enough if the keys are wrong.

### Validate the scorer before invoking a model

`E.gold_json` serializes the annotations as `{key: {"label": choice, "evidence": spans}}`, selecting keys present in the gold record. The notebook's `E.check_scorer` call checks that this reference is bare JSON and scores it against itself. This equivalent excerpt exposes the check without redefining the helper:

```python
from contractnli_scorer import score_record

_probe = E.gold_json(EVAL_DOCS[0], label_keys)
assert _probe.lstrip().startswith("{"), f"gold is not bare JSON: {_probe[:60]!r}"

_perfect = score_record({"id": "check", "model_response": _probe,
                         "reference_answer": _probe}, 0)
_flat = {m["name"]: m["value"] for m in _perfect["metrics_list"]}
assert _flat["label_correct"] == 1.0, f"scoring gold against itself gave {_flat}"
print("scorer self-check passed: gold scored against itself is 1.0")
```

This validates the reference handoff and label scoring on one contract, not learned quality or complete schema validity. The scorer accepts the top-level `reference_answer` only when its string begins with `{` after whitespace; wrapping the reference in commentary can silently produce zero content scores even if the model output parses. Keep `model_response` and `reference_answer` separate as the notebook does.

## 3. Generate and score endpoint predictions

### Start with one contract

Setup defaults are `RECORDS = 123`, `WORKERS = 8`, `BASELINES = "baselines"`, and both run switches `False`. To measure your deployed model, set `RUN_ENDPOINT_EVAL = True` in Setup and rerun the dependent cells. The runtime client uses a 900-second read timeout and `retries={"max_attempts": 0}`. This is not the deployment smoke test's timeout/retry configuration.

The utility takes clients and model identity explicitly instead of reading notebook globals. Its request implementation is:

```python
def ask_endpoint(doc, *, runtime, endpoint_name, served_model_name, labels, max_tokens=700):
    """One contract to the endpoint, returning the raw generated text."""
    payload = {
        "model": served_model_name,
        "messages": C.build_messages(doc, labels),
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": C.CHAT_TEMPLATE_KWARGS,
    }
    r = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                ContentType="application/json",
                                Body=json.dumps(payload))
    return json.loads(r["Body"].read())["choices"][0]["message"]["content"]
```

Do not redefine it in the notebook. The guarded smoke-test cell calls:

```python
if RUN_ENDPOINT_EVAL:
    _sample = E.ask_endpoint(
        EVAL_DOCS[0], runtime=runtime, endpoint_name=ENDPOINT_NAME,
        served_model_name=SERVED_MODEL_NAME, labels=labels)
```

`C.build_messages(doc, labels)` supplies only the system instruction/checklist and user contract. It does not append the gold assistant answer. `C.CHAT_TEMPLATE_KWARGS` is `{"enable_thinking": False}`, preserving the reasoning-off setting used in training. Evaluation sends string-valued message content; deployment's smoke test wraps the same text in typed text-content blocks. Evaluation's output budget is 700 tokens, whereas deployment defaults to 4,000. A successful smoke test therefore does not prove that every test answer fits the evaluation budget.

When enabled, the next lines invoke `EVAL_DOCS[0]`, print elapsed seconds and character count, and display the first 400 characters. This call is not caught by the full-pass error handler, so resolve a configuration error here before continuing. Inspect the complete `_sample` when checking all keys and citations, not just the printed prefix. The first document is invoked again during the full pass; the sample is not cached.

### Collect predictions without losing failed requests

`score_all` uses `ThreadPoolExecutor(max_workers=WORKERS)` to issue bounded concurrent requests. Its inner `one` function catches invocation/response-extraction exceptions, prints the record index and exception class, and returns an empty string for that record. Predictions are stored by enumeration index, then paired with the original documents in order. `executor.map` yields in input order, so a slow early request can delay progress-bar updates even when later requests have finished.

The scoring loop inside `score_all` is shown below. It runs after inference and retains raw generated text plus the four flattened metrics:

```python
rows = []
for i, d in enumerate(docs):
    scored = score_record({"id": str(i), "model_response": gens[i],
                           "reference_answer": gold_json(d, list(labels))}, i)
    flat = {m["name"]: m["value"] for m in scored["metrics_list"]}
    rows.append({"index": i, "generated": gens[i], **flat})
```

This excerpt uses `docs` and `gens` from the enclosing utility function; call `E.score_all(...)` from the notebook rather than running the excerpt in isolation. An empty response fails parsing and scores zero on all four metrics, remaining in the summary denominator. Parse failures therefore lower measured quality instead of disappearing from the evaluated population. The function does not distinguish an invocation failure from an empty model answer in its returned rows, retain the full error message, or store per-record latency, token usage, finish reason, gold text, or artifact identity. Scoring itself is outside the invocation `try` block, so malformed values that make the scorer raise can still interrupt the pass.

The full-pass cell makes the execution choice explicit:

```python
if RUN_ENDPOINT_EVAL:
    tuned_rows = E.score_all(
        EVAL_DOCS, runtime=runtime, endpoint_name=ENDPOINT_NAME,
        served_model_name=SERVED_MODEL_NAME, labels=labels, workers=WORKERS)
else:
    tuned_rows = E.load_rows("tuned_reference_rows.json", BASELINES)
tuned = E.summarise(tuned_rows)
```

Reference-only analysis summarizes the stored metric fields without re-scoring the saved generations. Both branches call `E.summarise`, implemented as:

```python
def summarise(rows):
    names = ("label_correct", "evidence_f1", "contradiction_correct", "json_valid")
    return {n: round(100 * statistics.mean(r[n] for r in rows), 2) for n in names}
```

The new live-row shape has six fields: `index`, `generated`, `label_correct`, `evidence_f1`, `contradiction_correct`, and `json_valid`. Shipped base and tuned rows also contain `reached_json`, but live `score_all` does not compute that field. A progress message such as `<count> contracts in <minutes> min` and four percentage-valued summary lines are expected output shapes, not a promised runtime or score.

**Checkpoint:** For a full run, verify 123 rows, indices `0` through `122`, nonempty generated answers where expected, and the number of zero `json_valid` values. Inspect printed errors and representative malformed answers before interpreting any comparison chart. Reducing `RECORDS` does not slice the shipped tuned rows, base/frontier summaries, or stored judge results, and chart titles retain hard-coded or configured counts. Restore the full dataset for reference comparisons. A live judge indexes `EVAL_DOCS` using stored row indices, so an undersized `EVAL_DOCS` can also produce `IndexError`.

## Performance measurement: latency is not TTFT

Quality scoring and serving performance are separate experiments. The optional **Streaming performance microbenchmark** in notebook 4 calls [inference_performance.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/inference_performance.py). Its `RUN_PERFORMANCE_BENCHMARK = False` switch is independent of the two quality/judge switches. Enable it only against your intended running endpoint, with permission for `sagemaker:InvokeEndpointWithResponseStream`. A fresh performance measurement does not make shipped quality results fresh.

| Quantity                                  | Current measurement boundary                                      | Interpretation                                                                                                                                       |
| ----------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Client full-response latency, `elapsed_s` | Immediately before streaming invocation through stream exhaustion | Includes network, queueing, prefill, generation, and response consumption; not just GPU time                                                         |
| Time to first generated content, `ttft_s` | Invocation start to first nonempty `delta.content`                | Excludes role-only events, reasoning-only events, usage metadata, and `[DONE]` as stopping triggers; a content chunk can contain more than one token |
| Output token rate, `output_tokens_per_s`  | Final `usage.completion_tokens / elapsed_s`                       | End-to-end token rate including prefill/network time, not decode-only throughput; absent final usage yields no token rate                            |
| Measured wall time                        | Start to finish of the measured batch                             | Use for fleet compute allocation; do not sum overlapping concurrent request durations                                                                |

The helper calls SageMaker's streaming API with `stream: true` and `stream_options: {"include_usage": true}`, preserving the shared message builder and `enable_thinking: false`. It reassembles server-sent events across arbitrary payload boundaries, closes the stream, and propagates API/stream failures rather than inventing timings. Confirm support in your serving image when changing versions. A non-streaming `invoke_endpoint` duration, including the deployment smoke test, cannot measure TTFT.

Notebook defaults are five shortest serialized contract requests, one separate serial warmup request, concurrency one, a 32 KiB serialized-request cap, and 256 maximum output tokens. The helper does not truncate input to fit the cap; it rejects oversized requests before invocation. A byte cap is not a tokenizer context limit. Warmup and measured requests are separate populations, with their own sample counts and summaries. The capped answers are a performance probe, not complete 17-item quality evaluations; inspect `finish_reason`, especially `length`.

For each timing or token-rate metric, the helper reports non-null sample count and linearly interpolated p50/p95 at rank `(n - 1) * percentile / 100` in sorted values. A p95 based on five samples is exploratory, not evidence of a production tail-latency guarantee. Short-input selection, warmed caches, client location, concurrency, output length, and other endpoint traffic all affect results. Retain warmup separately rather than quietly dropping it from the record.

**Activity:** run the bounded benchmark only if time and cost allow, then compare one sample's `elapsed_s`, `ttft_s`, usage, and finish reason. Explain why the two times differ. Save the `performance` object together with sample document IDs/order, selection rule, prompt bytes and token usage where available, output cap, template/decoding settings, warmup count, concurrency, failures, UTC measurement time, client location, account/Region, instance type/count, image, endpoint, and artifact identity. The helper retains sample results and timing summaries in memory; it does not automatically capture or persist that complete provenance manifest. Inspect endpoint/CloudWatch screens using the [deployment progress activity](/03-training-jobs/03-deployment/#use-provisioning-time-to-inspect-the-endpoint), without conflating service metrics with these client timings.

### Estimate serving compute without inventing a price

The optional cost cell leaves `SAGEMAKER_HOURLY_USD`, `PRICE_INSTANCE_TYPE`, `PRICE_SOURCE`, and `SERVING_LIFETIME_SECONDS` unset. `PRICE_REGION` defaults to the notebook Region and `SERVING_INSTANCE_COUNT` to one; verify both against the actual endpoint. The cell checks that the price Region matches the notebook Region and that instance type and source are supplied, but it does not fetch prices or verify deployment capacity for you. Fill them from the actual deployment and [SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/), recording Region, currency, lookup date, pricing model, and the pricing URL. Use the SageMaker real-time hosting rate, not an EC2 instance rate. `compute_cost` uses `seconds / 3600 * hourly_rate * instance_count`.

Deployment now uses [Instance Pools](/03-training-jobs/03-deployment/#instance-pools). `ml.g5.xlarge` is the preferred pool, not proof of the allocated type. Inspect the endpoint's actual pool types and current counts before setting `PRICE_INSTANCE_TYPE`, its rate, and `SERVING_INSTANCE_COUNT`. The current deployment requests one instance total; if you scale to a mixed fleet, the single-rate calculation cannot represent different pool prices. Sum `seconds / 3600 * hourly_rate * instance_count` separately for each allocated type and each interval with stable counts. Record that pool composition with performance measurements because fallback hardware can change latency and throughput, not just cost.

The measured-batch allocation and warmup allocation are reported separately. For a fixed fleet, total serving compute is based on the full billed provisioned lifetime, including idle time, not only benchmark seconds. Dividing the measured allocation by completed requests is not a per-request invoice or a complete application cost. Do not add overlapping batch allocations to the full lifetime total. Storage, transfer, Studio, MLflow, training, judging, taxes, and pricing adjustments remain separate; the helper is not billing reconciliation.

## 4. Understand exactly what the statistical metrics mean

[contractnli_scorer.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/contractnli_scorer.py) is called locally for each generated contract. Although the file also provides a Lambda wrapper, the notebook does not deploy or invoke it. Scoring is deterministic for fixed prediction/reference strings; inference and endpoint hosting are not free just because the comparison code makes no service call.

| Field                    | Per-contract calculation                                                 | Important limitation                                                                                          |
| ------------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `label_correct`          | Correct verdicts divided by reference items                              | Predicted label spelling is normalized for case and spaces; missing keys are wrong and extra keys are ignored |
| `evidence_f1`            | F1 from pooled span-set counts over items with nonempty gold evidence    | Computed within each contract, independently of whether the label is correct                                  |
| `contradiction_correct`  | Correct verdicts among gold `Contradiction` items divided by their count | Recall-like, not F1; returns zero for a contract with no gold contradictions                                  |
| `json_valid`             | One if the tolerant parser extracts a dictionary, otherwise zero         | Does not validate all checklist keys, entry types, allowed labels, or citation bounds                         |
| `aggregate_reward_score` | `0.6 * label_correct + 0.4 * evidence_f1`                                | Returned by `score_record`, but discarded by `score_all`; not plotted or averaged by this notebook            |

::alert[The notebook labels `contradiction_correct` as contradiction correctness. The helper computes neither contradiction precision nor contradiction F1. Read that column as the mean per-contract fraction of gold contradictions correctly identified, including zero for contracts with no gold contradictions. Historical outputs labeled `contra-F1` used the same recall-like field. Likewise, `json_valid` is not the same check as the judge's `ChecklistCompleteness`.]{type="warning"}

### Evidence overlap and aggregation

For a single contract, let \(G_i\) and \(P_i\) be the gold and predicted evidence sets for checklist item \(i\). Include only items for which \(G_i\) is nonempty:

```text
For each item i with nonempty gold evidence G_i:
TP = sum(size of intersection(G_i, P_i))
FP = sum(size of P_i minus G_i)
FN = sum(size of G_i minus P_i)
```

\[
P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}
\]

The helper uses zero when a denominator is zero. It converts accepted integer-like span values into sets, so repeated citations do not earn extra credit. Counts are accumulated separately for each item; identical span IDs in different items or documents are not merged into one global set.

For example, suppose two gold-evidence items have gold sets `{3, 4}` and `{5}`, and predictions cite `{4, 9}` and `{5}`. Then `TP=2`, `FP=1`, and `FN=1`, giving precision, recall, and evidence F1 of \(2/3\). If all three labels in the earlier synthetic schema were correct, the aggregate would be \(0.6\times1+0.4\times(2/3)\), rounded to four decimals. These are teaching-fixture arithmetic, not empirical model results. The following offline check exercises the actual scorer and utility without any model call:

```python
import json
import evaluation_utils as E
from contractnli_scorer import score_record

gold = {
    "nda-5": {"label": "Contradiction", "evidence": [3, 4]},
    "nda-4": {"label": "Entailment", "evidence": [5]},
    "nda-1": {"label": "NotMentioned", "evidence": []},
}
prediction = json.loads(json.dumps(gold))
prediction["nda-5"]["evidence"] = [4, 9]
scored = score_record({"model_response": json.dumps(prediction),
                       "reference_answer": json.dumps(gold)})
flat = {m["name"]: m["value"] for m in scored["metrics_list"]}
assert flat == {"label_correct": 1.0, "evidence_f1": 0.6667,
                "contradiction_correct": 1.0, "json_valid": 1.0}
assert scored["aggregate_reward_score"] == 0.8667
assert E.summarise([flat])["evidence_f1"] == 66.67
```

`score_record` rounds each metric to four decimals. `summarise` takes the arithmetic mean of those per-contract values, multiplies by 100, and rounds to two decimals:

```text
summary(metric) = round(100 * sum(per_contract_metric) / number_of_contracts, 2)
```

This is **mean per-contract evidence F1**, not corpus micro F1 obtained by pooling all contracts' counts before division. Contracts receive equal weight regardless of how many evidence spans they contain. The helper's introductory comment about an older notebook doing corpus micro scoring does not describe the current `summarise` implementation. Contradiction scores are also averaged by contract, including zero for contracts with no gold contradictions; they are not corpus contradiction recall. With exactly 17 reference items per contract, mean label accuracy has the same unrounded weighting as accuracy over all 2,091 decisions, apart from the helper's intermediate rounding.

### Parsing is not schema validation

`parse_answer` removes closed `<think>...</think>` blocks, accepts a fenced block, scans for a balanced JSON object, and tolerates a doubled opening brace. It can therefore count a dictionary surrounded by commentary as valid even though the application asks for JSON only. A parsed empty dictionary can have `json_valid = 1` while receiving zero content scores. Unparseable or truncated output receives zero content scores against the valid reference; those zeros are part of the reported accuracy, not measurements over successfully parsed answers only.

Before treating these scores as a release gate, separately check exact keys, object-valued entries, canonical labels, integer citation types, valid span IDs, and the `NotMentioned` empty-evidence rule. The helper ignores extra keys, can accept a label string where an item dictionary was expected, and normalizes some numeric strings. It does not receive the document's span inventory and cannot check citation bounds. Gold-empty items are excluded from evidence F1, so invented citations on those items are not penalized there. Conversely, an exact answer for a contract with no gold evidence gets zero evidence F1, not one.

Offline synthetic checks also expose remaining parser edge cases: escaped quotes with braces inside JSON strings can confuse the scanner, and an evidence string such as `"--1"` can pass the integer-like filter but raise `ValueError` when converted. `score_all` catches invocation errors, not exceptions from its later scoring loop, so those malformed values can interrupt the pass rather than yield a zero-valued row. These are helper limitations, not new measurements of model output.

Correct citation overlap can earn credit for a wrong verdict because evidence scoring is independent of labels. The score is an annotation-match statistic, not an independent legal judgment. A high deterministic score is useful evidence, but it is not proof that the answer is safe, complete, or immune to gaming.

## 5. Compare your run with the shipped references

The baseline comparison cell calls `E.load_precomputed` for `base`, `frontier`, and `tuned_reference`, loading their respective `_metrics.json` files before passing all three plus `tuned` to `E.compare_models`. If endpoint evaluation ran, the final row shows your percentage-point differences from that tuned reference. The chart displays label accuracy, evidence F1, and contradiction correctness; JSON validity is shown in the table but not that chart. The table uses the frontier file's model identifier, while the chart hard-codes “Sonnet 5” and “123 held-out contracts”; those labels do not adapt if you change files or population.

The following values are **stored reference results supplied with the lab**, not measurements of your endpoint. Their source files are [base_metrics.json](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/baselines/base_metrics.json), [tuned_reference_metrics.json](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/baselines/tuned_reference_metrics.json), and [frontier_metrics.json](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/baselines/frontier_metrics.json). All three declare 123 records.

| Stored reference                                          | Label accuracy (%) | Mean evidence F1 (%) | Mean `contradiction_correct` (%) | Parsed JSON (%) |
| --------------------------------------------------------- | ------------------ | -------------------- | -------------------------------- | --------------- |
| Untuned Nemotron 3 Nano 4B                                | 13.63              | 7.76                 | 3.79                             | 30.08           |
| Fine-tuned reference                                      | 87.90              | 76.60                | 70.12                            | 100.00          |
| Frontier file, labeled `global.anthropic.claude-sonnet-5` | 84.08              | 68.63                | 62.26                            | 100.00          |

The tuned reference metadata describes a one-epoch checkpoint trained on `ml.g5.2xlarge` and served on `ml.g5.xlarge`. It is not an artifact-identity check for your endpoint. The base/tuned metadata describes reasoning-disabled chat-template prompts; the frontier metadata describes plain messages without that template and a temperature that was not settable for its recorded model. These are useful task-specific reference comparisons, not identical serving/generation conditions or a general ranking of models. Reading the frontier file does not verify that identifier's availability in your account.

Inspect the underlying row files when explaining failures. For example, low parsed-JSON coverage can dominate the untuned summary because malformed answers contribute zeros. The metadata field `reached_json` is not equivalent to successfully completing a parseable answer. No benchmark time, cost advantage, or narrow tolerance around the reference scores is guaranteed for a new training/deployment run.

## 6. Evaluate semantics with the implemented Bedrock judge

Exact span overlap can miss an alternative clause that supports the same verdict. The judge is intended to review that semantic distinction. This track supplies **precomputed inference responses** to Bedrock: SageMaker or the shipped files provide the answers, and Bedrock judges them without loading the 4B model. No Model Package is required.

### Configure the evaluator and inspect the rubrics

The notebook sets `EVALUATOR_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"`, `JUDGE_N = 30`, `JUDGE_PREFIX = "contractnli-judge-eval"`, and `JUDGE_OUTPUT = f"s3://{bucket_name}/{JUDGE_PREFIX}/output"`. The evaluator identifier is a different model from the frontier reference label. Preserve the configured inference-profile identifier rather than substituting a bare model ID; verify its access in your Region.

| Custom metric             | Declared rating scale | Intended question                                                                                             |
| ------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------- |
| `EvidenceSupportsVerdict` | 0 to 3                | Does the text of the cited clause support each verdict, even when its span number differs from the reference? |
| `CarveOutHandling`        | 0 to 3                | Does the answer handle exceptions and carve-outs correctly?                                                   |
| `ChecklistCompleteness`   | 0 or 1                | Is there a well-formed entry for every checklist key, with label/evidence and no outside commentary?          |

Each metric is built as a `customMetricDefinition` with instructions and a rating scale. The instructions interpolate `{{prompt}}`, `{{prediction}}`, and, for the first two metrics, `{{ground_truth}}`. A shared introduction tells the judge to ignore reasoning blocks and unwrap stringified-list responses. The reference judge file documents normalized returned scores on 0 to 1, including approximately `0.3333` and `0.6667` for the 0-to-3 rubrics. The notebook does not perform an additional normalization; it averages the returned `result` values.

::alert[Review two real limitations before spending on a new judge run. `build_judge_dataset` passes only `C.build_messages(doc, labels)[1]["content"]`: the contract's user turn, not the system turn containing the checklist hypotheses and formatting instructions. `ChecklistCompleteness` asks for keys “listed in the prompt” but receives neither that checklist nor a reference placeholder. Also, the `CarveOutHandling` rubric says an absolute statement is contradicted by a clause “with no exception for it” and criticizes `NotMentioned` when the contract is silent. That wording conflicts with the task's definition of silence and its exception guidance. These are source limitations, not missing registry integration; interpret both fresh and shipped judge scores cautiously.]{type="warning"}

### Build the precomputed-response dataset

The utility API is `E.build_judge_dataset(rows, name, *, s3, bucket_name, documents, labels, judge_prefix, n=30)`. It takes the first `n` rows and retrieves their documents by `index`. Here is its record-building portion, with the enclosing function's arguments and module-local `gold_json` in scope:

```python
records = []
for r in rows[:n]:
    doc = documents[r["index"]]
    records.append({
        # the judge has to look up cited span numbers, so it gets the user turn verbatim
        "prompt": C.build_messages(doc, labels)[1]["content"],
        "referenceResponse": gold_json(doc, list(labels)),
        "modelResponses": [{"response": r["generated"], "modelIdentifier": name}],
    })
```

This becomes one JSON object per line in S3. `prompt` is the numbered contract string, `referenceResponse` is the gold JSON encoded as a string, and `modelResponses` contains the generated string. `modelIdentifier` is the comparison label `base` or `tuned`, not an endpoint name or model ARN. It must match the job's `inferenceSourceIdentifier`.

The builder serializes the records and uploads them with `put_object` to `contractnli-judge-eval/input/{name}-{n}.jsonl` in `bucket_name`, returning the S3 URI. With defaults, the keys end in `base-30.jsonl` and `tuned-30.jsonl`. These names are fixed, so rerunning uploads can overwrite an earlier run's input. The prefix is also used elsewhere in the workshop; verify ownership and preserve provenance instead of assuming it belongs exclusively to this track. The filename uses requested `n` even if fewer rows were supplied, so check the printed actual record count.

### Launch and monitor two evaluation jobs

`launch_judge` calls `bedrock.create_evaluation_job` with `applicationType="ModelEvaluation"`, task type `General`, dataset name `ContractNLI`, the custom metrics, and the configured evaluator. The critical inference handoff is `inferenceConfig={"models": [{"precomputedInferenceSource": {"inferenceSourceIdentifier": name}}]}`. Its `outputDataConfig` points to `JUDGE_OUTPUT`, and each job name incorporates the model label, requested count, and current timestamp.

After the helper definitions, the live branch executes:

```python
JUDGE_FILE = f"{BASELINES}/judge_reference_metrics.json"
if RUN_JUDGE_EVAL:
    bedrock = boto3.client("bedrock", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    base_rows = E.load_rows("base_rows.json", BASELINES)
    arns = {}
    for name, rows in (("base", base_rows), ("tuned", tuned_rows)):
        uri = E.build_judge_dataset(
            rows, name, s3=s3, bucket_name=bucket_name, documents=EVAL_DOCS,
            labels=labels, judge_prefix=JUDGE_PREFIX, n=JUDGE_N)
        arns[name] = launch_judge(uri, name)
    E.wait_for_judge(arns, bedrock=bedrock)
    judge_scores = {name: E.read_judge_scores(
        arn, bedrock=bedrock, s3=s3, bucket_name=bucket_name)
        for name, arn in arns.items()}
else:
    judge_scores = E.load_rows("judge_reference_metrics.json", BASELINES)["scores"]
    print(f"loaded pre-computed judge results from {JUDGE_FILE}")
```

`JUDGE_FILE` is `baselines/judge_reference_metrics.json`. With the switch disabled, its stored `scores` dictionary is loaded even if Part 1 generated new endpoint answers. With the switch enabled, the two jobs judge base and tuned answers only, not frontier answers. The base upload/submission happens before the tuned upload/submission; if the second fails, the first job may already be running. Save each printed job ARN and inspect it rather than blindly rerunning the cell.

`wait_for_judge` polls every 60 seconds while status is `InProgress`, `Scheduled`, or `Stopping`. It prints terminal states and failure messages for `Failed`, but does not raise on failure or stop other jobs. Interrupting the notebook's wait does not cancel remote evaluation. The source estimates minutes for endpoint scoring and longer for judge startup/execution; treat those as planning estimates only. Judge token usage and endpoint instance time are separately billed.

`read_judge_scores` obtains the output S3 URI through `get_evaluation_job`, paginates its prefix, and reads `.jsonl` objects whose keys contain the job ID from the ARN. From each line it extracts `automatedEvaluationResult.scores`, appending `score.get("result")` under `metricName`. It assumes the output uses `bucket_name`, consistent with this notebook's configuration. It does not retain document identities or explanations in `judge_scores`; retain the raw S3 results for auditing.

### Read means together with coverage

The judge table computes each cell this way:

```python
vals = scores_by_model[name].get(metric, [])
got = [v for v in vals if v is not None]
mean = sum(got) / len(got) if got else float("nan")
row += f"{mean:.3f} ({len(got)}/{len(vals)})".rjust(22)
```

This is an excerpt from `judge_table`, not a separate evaluator. An output cell has the shape `<mean> (<non-null scores>/<returned score entries>)`. Numeric zero remains in the mean; `None` is excluded. If no numeric value is available, the mean is `nan`. For a normalized 0-to-3 rubric, a mean of 0.5 corresponds to an average rating of 1.5 out of 3, not “50% of contracts correct.”

Coverage matters because base and tuned means may describe different subsets. In addition, the denominator is the number of score entries actually found, not necessarily the number of submitted contracts. Missing whole records or missing metric entries are not padded with `None`. Compare every denominator with the actual uploaded count and inspect raw outputs before claiming full coverage. The reader flattens scores without retaining IDs, so it does not implement paired-contract filtering. A paired semantic comparison requires inspecting the raw record identities.

The judge chart shows mean scores on a 0-to-1 axis. Its code builds coverage strings but does not display them, so read the table alongside the chart. If all jobs return no scores, `judge_table` can fail while computing the maximum metric-name width; inspect job statuses and S3 outputs rather than treating an empty result as zero quality.

A high semantic rating with lower exact evidence F1 can suggest an acceptable alternate citation, but it does not prove an annotation error. Inspect the clauses and involve a qualified reviewer where appropriate, especially given the prompt/rubric limitations above. Contract and model text are untrusted inputs to the judge; instructions inside them can influence a model-based evaluator. The judge is a diagnostic aid, not authoritative legal validation.

## 7. Inspect errors and retain the actual outputs

After the charts, the notebook calls `E.show_error_directions(tuned_rows, EVAL_DOCS)`. It delegates counting to `E.error_directions` and prints at most six gold-to-predicted label confusions. It uses strict `json.loads`, not the scorer's tolerant parser, zips rows and documents by position, and only counts present item dictionaries with a nonempty label that differs from gold. Unparseable responses, missing keys, empty labels, and citation-only errors are omitted. Label spelling is not normalized here. This is a focused diagnostic, not a complete confusion matrix or failure inventory.

| Artifact or variable                                                                   | What the current notebook actually does                                                                               |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `baselines/base_metrics.json`, `frontier_metrics.json`, `tuned_reference_metrics.json` | Reads shipped summary metadata and scores; does not overwrite them                                                    |
| `baselines/tuned_reference_rows.json`                                                  | Reads saved tuned generations and metric fields when endpoint evaluation is disabled                                  |
| `baselines/base_rows.json`                                                             | Reads saved untuned generations for a live judge run                                                                  |
| `baselines/frontier_rows.json`                                                         | Available for inspection; not read by the notebook's comparison or judge cells                                        |
| `baselines/judge_reference_metrics.json`                                               | Reads saved judge scores when judge execution is disabled                                                             |
| `tuned_rows`, `tuned`, `judge_scores`, and live `arns`                                 | Kept in kernel memory; no new local results JSON, CSV, or Parquet file is written                                     |
| Optional `performance`                                                                 | Streaming sample results, usage/finish reason, separate warmup/measured summaries and wall times; also kept in memory |
| Printed tables, timings, errors, and Matplotlib charts                                 | Displayed in notebook output; charts are not saved with `savefig`                                                     |
| `s3://<bucket>/contractnli-judge-eval/input/{base,tuned}-30.jsonl`                     | Uploaded only for live judging, containing prompt, reference, and generated text                                      |
| `s3://<bucket>/contractnli-judge-eval/output`                                          | Configured prefix for managed judge results, read by job ID; exact object keys come from the service                  |

Save approved notebook outputs and export any additional experiment record explicitly before restarting the kernel; the notebook does not provide automatic run persistence. Retain the source revision, helper/config versions, raw dataset identity/order, switches, actual record counts, generation parameters, model/artifact/endpoint identities, raw generations, and judge ARNs/output locations. Do not overwrite shipped baselines to make your result look like the reference. Quality-scoring rows do not contain per-record performance data. The separate opt-in streaming benchmark supplies timing/usage samples and rate-based compute allocations only when run; retain its `performance` object separately and do not attach those measurements to all 123 quality records. Do not claim confidence intervals, ROUGE, BERTScore, evaluation MLflow logging, or invoice totals that this code does not produce.

### Troubleshooting checkpoints

| Symptom                                                    | What to check                                                                                                                                                                                                             |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Missing helper, baseline, or raw test file                 | Kernel working directory and complete source checkout; `CONTRACTNLI_DIR` versus the download destination; `ensure_dataset` checks for `train.json`, not the integrity of every split                                      |
| Scorer self-check fails or test list is empty              | Raw split structure, `RECORDS > 0`, checklist keys, and bare-JSON reference construction; do not proceed with a bad reference                                                                                             |
| Endpoint not found or a vLLM model-name error              | Region, reconstructed endpoint name, `InService` status, and exact `nemotron-contractnli` serving alias                                                                                                                   |
| Invocation errors, low JSON coverage, or truncated answers | Inspect `_sample`, raw generations, and endpoint logs; verify reasoning-off reaches the template, the 700-token output budget, input length versus the serving context limit, and concurrency before interpreting quality |
| Unexpectedly unchanged results                             | Check `RUN_ENDPOINT_EVAL`; confirm artifact lineage behind the stable endpoint; stored metrics are read rather than re-scored in reference mode                                                                           |
| `IndexError` when building judge data                      | Stored row indices must index the same full `EVAL_DOCS`; do not mix a short slice or reordered documents with full shipped rows                                                                                           |
| Judge access denied or throughput validation error         | Caller evaluation/PassRole permissions, Bedrock trust, S3/KMS permissions, and access to the configured evaluator inference profile                                                                                       |
| Failed/stopped judge or empty metric table                 | Inspect printed job ARNs, terminal statuses, failure messages, output prefix and job-ID filtering; the wait helper does not enforce successful completion                                                                 |
| Judge appears unrelated to your fresh model                | A disabled judge switch loads historical judgments; the live prompt omits the system checklist, and the carve-out rubric has contradictory wording                                                                        |
| Cleanup prints `skip`                                      | Inspect the exact error, resource ownership, permissions, and asynchronous deletion state; a skipped operation is not successful cleanup                                                                                  |

## 8. Clean up deliberately

The current deployment notebook leaves hosting running for evaluation. Hosting teardown is at the end of **`4-evaluation.ipynb`**, not `3-deployment.ipynb`. Inspect the reconstructed names and confirm ownership first, especially if you only loaded reference results. The cell attempts deletion in this dependency order:

```python
from sagemaker.core.resources import Endpoint, EndpointConfig, Model

stem = f"{MODEL_SLUG}-contractnli"
for label, fn in [
    ("endpoint", lambda: Endpoint.get(ENDPOINT_NAME).delete()),
    ("endpoint config", lambda: EndpointConfig.get(E.rname(stem, "-sft-cfg")).delete()),
    ("model", lambda: Model.get(E.rname(stem, "-sft-m")).delete()),
]:
    try:
        fn()
        print(f"deleted {label}")
    except Exception as error:
        print(f"skip {label}: {error}")
```

The loop has no explicit waiter between operations and catches all errors. Confirm that endpoint deletion finishes and investigate skipped configuration/Model deletions before considering teardown complete. An endpoint remains a separately billed hosting resource while you read references or wait for the judge. Bedrock judging needs only the uploaded responses, not a live endpoint, once inference is complete.

This cell does not stop Bedrock jobs, remove judge inputs/outputs, release a retained training warm pool, delete S3 training artifacts or logs, or stop Studio compute. Both tracks can share S3 prefixes, so remove only the objects and jobs you own after applying retention requirements. Never delete the default bucket wholesale. Complete [Clean Up](/04-cleanup/) for the broader resource review, using the current notebook filename and including the 4B journey's Bedrock jobs and judge S3 objects described here.

## Completion criteria

You should be able to state whether each displayed result is historical or freshly computed, identify the actual artifact behind a live endpoint, explain the four statistical fields and their aggregation, account for malformed outputs, and describe the judge's prompt, coverage, and rubric limitations. A successful run establishes measured behavior on its recorded test population, not general legal correctness, production readiness, frontier-model parity, or a guaranteed cost advantage. Preserve the run evidence and verify cleanup before leaving the lab.
