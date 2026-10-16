---
title: "Evaluation"
weight: 3
---

## What you will learn

You will distinguish schema compliance from task quality, calculate verdict and citation metrics, validate a self-contained custom scorer, launch registry-based evaluation for base and tuned models, and interpret a semantic judge without hiding missing judgments or mixing unrelated runs.

::alert[Open [01-serverless-workshop/3-evaluation.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/3-evaluation.ipynb) after training completes. This workflow uses the registered test dataset and Model Package. You do not need to create the real-time endpoint first.]{type="info"}

The notebook imports [evaluation_utils.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/evaluation_utils.py) as `E`. It retains the `CustomScorerEvaluator` and `LLMAsJudgeEvaluator` constructors and their `.evaluate()` submissions. The utility owns the frontier Converse requests/retries, local parsing and metrics, scorer checks, judge definitions, polling, S3 readers, diagnostics, plots, and JSON exports. Importing it creates no AWS clients or jobs, but running the notebook still performs live evaluation; there is no reference-only execution switch in this route. Do not substitute the 4B track's same-named helper.

| API                                            | What the notebook delegates                                                                       |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `E.model_package_group_name(BASE_MODEL_ID)`    | The same hash-truncated package-group name used by training and deployment                        |
| `E.evaluate_with_bedrock(...)`                 | Concurrent frontier generation, item-level scoring, and failure counts                            |
| `E.check_scorer`, `E.show_scorer_source`       | Paired sanity check and optional scorer inspection                                                |
| `E.build_judge_metrics`, `E.wait_for_pipeline` | Fresh rubric definitions and pipeline monitoring                                                  |
| `E.EvaluationResults`                          | Execution lookup, prompt checks, S3 result loading, comparisons, distributions, and local exports |

`contractnli_scorer.py` remains a separate self-contained upload. Moving orchestration support into `evaluation_utils.py` does not change the scorer's equations or make a strict schema validator part of evaluation.

## Define the evaluation contract before running it

The held-out set contains 123 contracts with 17 checklist decisions each. A prediction must assign a verdict and evidence to every key. Keep this test set separate from training and hyperparameter selection. If you reduce `EVAL_N`, record that change: it limits the notebook's local baseline, not automatically the full registered dataset consumed by managed evaluation.

The business risks differ. Incorrectly predicting `Entailment` can claim that a protection exists when the contract is silent. Missing a real contradiction can hide an exception. A correct verdict paired with an unrelated citation can still mislead a reviewer. Measure these dimensions separately before reducing them to a single score.

| Evaluation path        | Model input                                                     | Result source                                                    |
| ---------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| Local Bedrock baseline | System checklist and user contract through Converse             | Notebook score rows and `./eval_result` snapshots                |
| Managed custom scorer  | Registered test `query` and `response`, package plus base model | Pipeline S3 aggregates and per-record details                    |
| Managed LLM judge      | Generated answers plus task-specific rubrics                    | Bedrock evaluation scores and explanations under pipeline output |

The executable baseline identifier is `global.anthropic.claude-sonnet-4-6`; `BEDROCK_REGION` defaults to the SageMaker Region. Verify access before launching calls. The baseline uses `C.build_system(labels)` and `C.build_user(doc, no_think=False)`, whereas managed evaluation uses the single-string `query` ending with `/no_think`. These are different prompt presentations. Historical model names and benchmark tables elsewhere in the notebook are not evidence of your run.

### Run the local frontier pass

After the notebook creates the session, clients, `EVAL_DOCS`, and `results`, it submits the baseline through the utility:

```python
bedrock = boto3.client(
    "bedrock-runtime", region_name=BEDROCK_REGION,
    config=Config(read_timeout=300, retries={"total_max_attempts": 3}))
frontier_metrics, frontier_rows = E.evaluate_with_bedrock(
    EVAL_DOCS, FRONTIER, client=bedrock, labels=labels, workers=6)
results.save_json("frontier_metrics", frontier_metrics)
results.save_json("frontier_rows", frontier_rows)
```

`E.ask_bedrock` defaults to 4,000 output tokens and temperature zero, sends the checklist in Converse's `system` parameter, and concatenates text parts from the response. It permits five helper-level attempts, removes temperature if a service error mentions that field, and retries messages containing throttling, 429, 503, or timeout markers with exponential delays starting at two seconds and capped at 30. These attempts sit above the client's retry configuration; this is not a promise of five total network calls. Other errors are raised to the collecting function.

Six workers can complete out of order, so the utility returns each document's index with its parsed answer, fills an ordered prediction list, then scores in input order. An exhausted request or unparseable reply remains `None`; its gold checklist items stay in the denominator with missing predictions. `parse_failures` counts both categories at contract level, whereas `decisions` counts scored gold checklist items. The progress bar's `json_ok` counts parsed objects, including incomplete objects, not complete correct contracts. No raw frontier generations or request timings are returned.

## 1. Understand parsing and row construction

The local parser removes completed `<think>...</think>` blocks, looks inside a Markdown code fence if present, and scans braces to find a JSON object. The helper scorer has similar extraction logic and checks that the result is a dictionary. This is tolerant extraction, not strict JSON-only compliance: a reply with commentary can still be counted as parseable.

`E.score_contract` expands a prediction into one row per checklist key that also exists in the gold annotations. Each row contains the gold/predicted label and sets of gold/predicted evidence identifiers. Missing entries remain wrong rather than disappearing from the denominator. Evidence strings that look like integers are converted into integers, and repeated citations collapse in a set.

The managed `score_record` helper also normalizes label case and whitespace, while the notebook's local row scorer compares the label directly. Record this distinction if you compare their results. Neither parser is a complete production schema validator. Separately check exact key membership, valid label values, integer evidence, in-document span bounds, and empty citations for `NotMentioned`.

Two remaining edge cases were reproduced with synthetic inputs against the actual helpers offline: escaped quotes followed by braces inside a JSON string can confuse the brace scanner even when `json.loads` accepts the object, and an evidence value such as `"--1"` passes the integer-like filter but raises `ValueError` during conversion. The local frontier scorer can then interrupt the pass after generation; the uploaded scorer's Lambda wrapper instead returns an error. Tolerant extraction is not a guarantee that arbitrary malformed input becomes a zero score.

## 2. Calculate the statistical metrics

Let \(N\) be the number of scored checklist decisions. Label accuracy is \(\sum_i \mathbf{1}[\hat y_i=y_i]/N\). For a particular class, precision is \(TP/(TP+FP)\), recall is \(TP/(TP+FN)\), and \(F_1=2PR/(P+R)\). The helpers return zero when the relevant denominator is zero.

For evidence, each eligible checklist item has a gold set \(G_i\) and a predicted set \(P_i\). Accumulate \(TP=\sum_i|G_i\cap P_i|\), \(FP=\sum_i|P_i-G_i|\), and \(FN=\sum_i|G_i-P_i|\), then compute F1. Only items with non-empty gold evidence participate in this calculation.

A worked arithmetic example, not a model result: with gold `{3, 4}` and prediction `{4, 9}`, there is one true-positive citation, one false positive, and one false negative. Precision and recall are each 0.5, so evidence F1 is 0.5. This offline fixture exercises the utility's actual row constructor and percentage-valued aggregation:

```python
import evaluation_utils as E

doc = {"annotation_sets": [{"annotations": {
    "nda-1": {"choice": "Contradiction", "spans": [3, 4]}
}}]}
prediction = {"nda-1": {"label": "Contradiction", "evidence": [4, 9]}}
rows = E.score_contract(doc, prediction, ["nda-1"])
assert E.metrics(rows) == {
    "decisions": 1, "accuracy": 100.0, "contradiction_f1": 100.0,
    "evidence_f1": 50.0, "parse_failures": 0,
}
```

A correct label does not erase the wrong citation. Evidence is scored independently of label correctness, so correct spans can also earn overlap credit beside a wrong verdict. The fixture's values test arithmetic only; they are not quality measurements of a model.

| Managed per-contract metric | Definition and limitation                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `label_correct`             | Fraction of reference keys with the correct normalized label                                              |
| `evidence_f1`               | F1 from citation counts pooled within this contract, excluding gold-empty items                           |
| `contradiction_correct`     | Fraction of gold `Contradiction` items correctly labeled; zero if the contract has no gold contradictions |
| `json_valid`                | Whether the tolerant parser extracts a JSON object; not a full schema check                               |
| `aggregate_reward_score`    | `0.6 * label_correct + 0.4 * evidence_f1`, the chosen workshop weighting                                  |

The local `contradiction_f1` is one-vs-rest F1 and penalizes false-positive contradictions. Managed `contradiction_correct` is recall-like on gold contradictions and is **not** that F1. Averaging its zero values for contracts with no contradictions also changes the interpretation. Report the metric's actual name and denominator.

The managed pipeline averages per-contract scores rounded by `score_record` to four decimals. The local `E.metrics` returns percentages rounded to one decimal and instead pools all contracts' citation counts before calculating evidence F1. In general, `mean(F1 per contract)` is not `F1(pooled counts)`. The local function reports percentages and the managed scorer reports fractions. Divide the local numbers by 100 before placing both on a 0-to-1 plot, and do not rank close scores as if the aggregation methods were identical.

Because gold-empty items are excluded from evidence F1, invented citations on those items are not penalized by this evidence metric. Extra checklist keys also do not enter the reference-key loop. Add schema and unsupported-citation checks when designing a production gate. A deterministic formula can still be incomplete or reward a shortcut.

## 3. Guard against reference leakage

Read [contractnli_scorer.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/contractnli_scorer.py). The managed pipeline uploads this self-contained file; it cannot import functions defined only in your notebook. Its `model_response` field contains generated text. The payload's `response` field can contain the gold answer inherited from the test dataset. Scoring `response` as the prediction would compare the reference with itself.

The notebook calls `E.check_scorer(EVAL_DOCS[0])` before registration. The utility retains this paired check:

```python
import json
from contractnli_scorer import score_record

doc = EVAL_DOCS[0]
gold = {k: {"label": v["choice"], "evidence": list(v["spans"])}
        for k, v in C.gold_for(doc).items()}
reference = json.dumps(gold)
lazy = json.dumps({k: {"label": "NotMentioned", "evidence": []} for k in gold})

lazy_score = score_record({"id": "lazy", "model_response": lazy,
                           "reference_answer": reference})["aggregate_reward_score"]
gold_score = score_record({"id": "gold", "model_response": reference,
                           "reference_answer": reference})["aggregate_reward_score"]
assert gold_score == 1.0
assert lazy_score < gold_score
```

This check assumes the selected contract contains some gold evidence, as in the notebook example. For an entirely gold-empty synthetic contract, the helper's evidence F1 is zero even for an exact answer, so its aggregate would not be one. That edge case follows from the metric definition, not from reference leakage. Do not use an arbitrary threshold such as “every lazy answer must score below 0.5”; a document with many genuinely unmentioned items can legitimately exceed that threshold.

The helper returns an object with `id`, a numeric `aggregate_reward_score`, and `metrics_list`, whose entries contain `name`, `value`, and `type`. Those are result **fields**, not expected measured values. Preserve the individual metrics even when an aggregate is available.

## 4. Register the scorer and launch managed evaluation

Setup uses the same MLflow resolver as training and passes the resolved `mlflow_app_arn` explicitly to both managed evaluators. Save training's printed ARN into `MLFLOW_APP_ARN` in `config.py` to ensure evaluation reuses that exact app; an empty value first discovers the tagged stack app, then the named workshop app, before SDK fallback. Multiple matching stack apps require an explicit ARN; SDK fallback should not be assumed to select training's app in a shared account. An invalid explicit ARN fails without replacement. To inspect associated runs, use Studio's **MLflow** application and select the app matching the resolved ARN; retain each pipeline execution ARN rather than assuming the latest run belongs to this evaluation. No Tracking Server or domain association is required. Setup choices and permissions are covered in [MLflow App setup](/01-prerequisites/3-sagemaker/#7-reuse-or-create-the-workshop-mlflow-app).

Run the registration cell, which retrieves `contractnli-scorer` or creates it with `Evaluator.create(type=REWARD_FUNCTION, source="contractnli_scorer.py", ...)`. If it reuses an evaluator, verify its version and source correspond to the helper you inspected; reusing a name does not upload a changed file. The same cell retrieves `contractnli-nda-review-test` and the newest package in the derived Model Package Group. Check that package's training lineage before submission.

```python
from sagemaker.train.evaluate import CustomScorerEvaluator

SCORER_OUTPUT = f"s3://{BUCKET}/contractnli-scorer-eval"
scorer_eval = CustomScorerEvaluator(
    evaluator=scorer,
    dataset=test_dataset,
    model=model_package_arn,
    model_package_group=model_package_group_name,
    s3_output_path=SCORER_OUTPUT,
    evaluate_base_model=True,
    mlflow_resource_arn=mlflow_app_arn,
    sagemaker_session=sess,
    role=role,
)
scorer_eval.hyperparameters.max_new_tokens = 8192
scorer_eval.hyperparameters.max_model_len = 24000
scorer_execution = scorer_eval.evaluate()
print(scorer_execution.arn)
```

`max_new_tokens` budgets generated output; `max_model_len` constrains the inference context. Neither changes the training sequence cap. Inspect the resolved temperature and other generation defaults printed by the notebook. A large output budget does not guarantee complete JSON if the input is too long or generation does not terminate as intended.

The notebook calls `E.wait_for_pipeline(sm, scorer_execution.arn)`; its default 60-second polling behavior is equivalent to:

```python
import time

while True:
    status = sm.describe_pipeline_execution(
        PipelineExecutionArn=scorer_execution.arn)["PipelineExecutionStatus"]
    print(status)
    if status != "Executing":
        break
    time.sleep(60)
```

Continue to result inspection only after verifying success, not merely because the polling loop exited. Inspect failed pipeline steps and the exact error before resubmission. Evaluation inference and managed execution incur charges even though the scoring arithmetic itself is local deterministic computation.

## 5. Inspect what each model actually received

The notebook reads detail Parquet files for `EvaluateBaseModel` and `EvaluateCustomModel`. Its `full_prompt` column captures the evaluation container's rendered turns. Confirm `CHECKLIST:` appears for **both** paths, and inspect the contract, output-format instruction, and reasoning control. The actual test records already put the instruction inside `query`; do not follow older notebook prose that describes a separate `system` column as if it were the current schema.

Keep `scorer_execution.arn` with every result. The helper `find_s3_keys` attempts to filter by execution ID but falls back to all matching objects if that filter finds nothing. Its “latest” fallback can therefore load an older run. Similarly, `last_successful_execution` searches by pipeline kind, not your exact package. Verify the S3 key and execution provenance explicitly before trusting charts.

| Result check                                   | Why it matters                                                                                  |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `byoc_failure_count` is reviewed               | Scorer failures can reduce the scored population without making every average obviously invalid |
| Detail-row count matches the intended test set | Missing contracts weaken coverage                                                               |
| Per-record means reproduce aggregate values    | Helps detect the wrong detail file or scale                                                     |
| Model/package and execution IDs are recorded   | Prevents labeling a prior experiment as the current run                                         |
| Parser failures are reported separately        | Separates unusable output from wrong decisions in usable output                                 |

The result object is created once with explicit dependencies and output prefixes:

```python
SCORER_OUTPUT = f"s3://{BUCKET}/contractnli-scorer-eval"
JUDGE_OUTPUT = f"s3://{BUCKET}/contractnli-judge-eval"
results = E.EvaluationResults(
    s3=s3, sm=sm, bucket=BUCKET, documents=EVAL_DOCS, checklist=CHECKLIST,
    base_model_id=base_model_id, frontier_model=FRONTIER,
    scorer_output=SCORER_OUTPUT, judge_output=JUDGE_OUTPUT)
SCORER_EXECUTION = results.resolve_execution(
    globals().get("scorer_execution"), "CustomScorerEvaluation")
results.verify_prompts(SCORER_EXECUTION)
base_scored, tuned_scored = results.scorer_results(SCORER_EXECUTION)
results.compare_models(base_scored, frontier_metrics, tuned_scored)
base_details, tuned_details = results.scorer_distributions(
    base_scored, tuned_scored, SCORER_EXECUTION)
```

`verify_prompts` checks the first `full_prompt` row for each model, not every record. `scorer_distributions` unpacks `custom_lambda_metric` from the details, asserts that the four column means reproduce aggregates within `1e-6`, and plots per-contract distributions. This mean check does not assert equal row counts, identical document IDs, or complete dataset coverage. Check those separately. Restart recovery locates an execution; it does not reload `frontier_metrics` or reconstruct all notebook state automatically.

| Exact local file, relative to the serverless notebook directory     | Saved payload                                                                                                                                   |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `./eval_result/frontier_metrics.json`                               | `decisions`, percentage `accuracy`, `contradiction_f1`, `evidence_f1`, and contract-level `parse_failures`                                      |
| `./eval_result/frontier_rows.json`                                  | One row per item: `key`, `gold`, `pred`, `gold_ev`, `pred_ev`; evidence sets become sorted lists                                                |
| `./eval_result/scorer_base.json`, `./eval_result/scorer_tuned.json` | Managed aggregate dictionaries, including any service-reported failure fields                                                                   |
| `./eval_result/summary.json`                                        | `eval_contracts`, `checklist_items`, and `models` entries `base`, `frontier`, `fine_tuned`, each with `model`, `how`, `accuracy`, `evidence_f1` |
| `./eval_result/judge_base.json`, `./eval_result/judge_tuned.json`   | Flattened `metric`/`score` rows from the later judge stage; missing scores serialize as JSON `null`                                             |

`save_json` creates `./eval_result` and overwrites these fixed filenames on reruns. The exports do not include execution ARNs, document IDs, raw generations, or a full provenance manifest; `summary.json` identifies the tuned model by base ID rather than package ARN. Retain those identities separately and archive wanted snapshots before rerunning. The managed aggregates/details remain under `SCORER_OUTPUT`, and judge results under `JUDGE_OUTPUT`; their exact service-generated keys must be inspected. Charts are displayed, not saved as image files. The hard-coded historical comparison text still mentions a different model size and frontier name; use returned metrics and configured identities, not that printout, for your report.

## 6. Add a semantic judge

Exact citation overlap cannot recognize two different clauses that state the same relevant restriction. The judge reads the clause text and asks whether it supports the verdict, supplementing rather than replacing the expert annotation comparison. A disagreement is a review candidate, not automatic proof that the annotation or model is wrong.

| Custom rubric             | Scale declared by the notebook | Review question                                                    |
| ------------------------- | ------------------------------ | ------------------------------------------------------------------ |
| `EvidenceSupportsVerdict` | 0 to 3                         | Do the cited clauses actually justify the committed verdicts?      |
| `CarveOutHandling`        | 0 to 3                         | Were exceptions and carve-outs interpreted appropriately?          |
| `ChecklistCompleteness`   | 0 or 1                         | Are all required entries and the requested JSON structure present? |

Inspect `E.build_judge_metrics()` before submission. Every call returns fresh definitions that you can inspect or edit without sharing mutable rubric state. All three prompts tell the judge to focus on final JSON, ignore completed reasoning preambles, and unwrap stringified-list responses. The first two interpolate `{{prompt}}`, `{{prediction}}`, and `{{ground_truth}}`; completeness uses only prompt and prediction. The 0-to-3 scales run from unrelated citations/ignored exceptions to every verdict supported/every exception handled; completeness is binary.

The `CarveOutHandling` text says an absolute statement is contradicted by a clause “with no exception for it” and criticizes `NotMentioned` when the contract is silent. That conflicts with the task's silence and exception definitions, not just the metric label. Review the returned definitions and judge explanations before treating this rubric as evidence of legal correctness. Contract text and model output are untrusted data, including for a judge; instructions embedded inside them should not control grading.

This route delegates judge-input construction to the managed evaluator with the registered `query`/`response` dataset. The registered query contains both contract and checklist as a single string; there is no local user-turn-only judge dataset builder here. `verify_prompts` inspects the statistical pipeline's inference prompts, not the actual judge payload. Inspect managed judge inputs and explanations to confirm checklist visibility rather than applying the 4B route's prompt behavior to this route. The notebook sets 8,192/24,000 generation overrides on the statistical evaluator only; the separate judge evaluator does not inherit those assignments.

::alert[The managed judge needs Bedrock service-role trust **and** identity permissions, including the applicable evaluation operations, model invocation, S3 access, and role passing. Adding `bedrock.amazonaws.com` to a trust policy does not grant `bedrock:CreateEvaluationJob`. Review [account setup](/01-prerequisites/2-account/) before starting another billed pipeline.]{type="warning"}

The notebook obtains `custom_metrics = E.build_judge_metrics()`, serializes it with `json.dumps`, and passes the resulting **JSON string**, not a Python list:

```python
from sagemaker.train.evaluate import LLMAsJudgeEvaluator

EVALUATOR_MODEL = "amazon.nova-pro-v1:0"
custom_metrics_json = json.dumps(custom_metrics)
JUDGE_OUTPUT = f"s3://{BUCKET}/contractnli-judge-eval"
judge = LLMAsJudgeEvaluator(
    model=model_package_arn,
    model_package_group=model_package_group_name,
    evaluator_model=EVALUATOR_MODEL,
    dataset=test_dataset,
    custom_metrics=custom_metrics_json,
    s3_output_path=JUDGE_OUTPUT,
    evaluate_base_model=True,
    mlflow_resource_arn=mlflow_app_arn,
    sagemaker_session=sess,
)
judge_execution = judge.evaluate()
print(judge_execution.arn)
```

After verifying pipeline success, use `judge_execution.show_results(limit=5, offset=0, show_explanations=True)`. The notebook reads per-record JSONL entries from `automatedEvaluationResult.scores`, with `metricName` and `result`. It treats returned scores as normalized values and preserves `null` results as missing rather than replacing them with zero. Inspect actual payloads to verify the scale before plotting.

The notebook calls `results.judge_results(JUDGE_EXECUTION)` after resolving the execution with `results.resolve_execution(..., "LLMAJEvaluation")`. It selects the newest matching `_output.jsonl` object for each of `base-llmaj-eval` and `custom-llmaj-eval`, flattens scores, draws a coverage-annotated bar chart and a radar chart, and saves both judge snapshots. It does not merge every matching output shard or retain document IDs/explanations in those snapshots. Keep the raw S3 records for audit and use `show_results(..., show_explanations=True)` to inspect reasoning.

Coverage on the bar chart is non-null scores divided by returned score entries for each rubric/model, not an independently verified submitted-contract count. Missing whole records or metric entries are not padded. Compare each denominator to the actual test population; neither equal means nor the radar chart establishes complete coverage. A mean over one subset cannot be subtracted from a mean over a different subset and called a paired improvement. Pairing requires raw record identities, which the flattened reader discards. No confidence intervals or significance tests are computed. Keep judge model identity, rubric text, and execution ID with the result because model-based judgments can vary.

## Review and handoff

Inspect error directions, not only averages: `NotMentioned` to `Entailment`, missed contradictions, invalid citation IDs, incomplete checklists, and alternate-but-plausible evidence should be reviewed separately. Establish release thresholds for your own application before looking at the test results; this workshop supplies no guaranteed improvement, frontier parity, runtime, or cost target.

| Symptom                                | Next check                                                                                                                                                                                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Perfect scores for obviously poor text | Verify `model_response` versus reference fields with the paired sanity check                                                                                                                                              |
| Empty or old result files              | Confirm successful execution and exact S3 keys; reject the helper's cross-run fallback                                                                                                                                    |
| Low parse rate                         | Inspect raw text, finish behavior, actual prompt, and generation budget                                                                                                                                                   |
| Comparison raises `ZeroDivisionError`  | `compare_models` and `plot_statistical_comparison` divide by base `json_valid` in their low-coverage annotations; zero validity is not guarded. Inspect scorer results directly rather than reporting a conditional score |
| High parse rate, poor evidence         | Review cited clauses and label errors rather than tuning formatting alone                                                                                                                                                 |
| Judge access denied after inference    | Check the Bedrock evaluation identity policy in addition to trust                                                                                                                                                         |
| Unequal judge coverage                 | Report missingness and use a paired subset before comparing means                                                                                                                                                         |

Retain the package ARN, data versions, scorer version, generation configuration, execution IDs, and coverage with your findings. Continue to [Deployment](/02-serverless/04-deployment/) for an application-facing smoke test, or [Clean Up](/04-cleanup/) if you will not host the model.
