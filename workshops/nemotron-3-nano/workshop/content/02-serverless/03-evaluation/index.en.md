---
title: "Evaluation"
weight: 3
---

## What you will learn

You will distinguish schema compliance from task quality, calculate verdict and citation metrics, validate a self-contained custom scorer, launch registry-based evaluation for base and tuned models, and interpret a semantic judge without hiding missing judgments or mixing unrelated runs.

::alert[Open [01-serverless-workshop/3-evaluation.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/3-evaluation.ipynb) after training completes. This workflow uses the registered test dataset and Model Package. You do not need to create the real-time endpoint first.]{type="info"}

## Define the evaluation contract before running it

The held-out set contains 123 contracts with 17 checklist decisions each. A prediction must assign a verdict and evidence to every key. Keep this test set separate from training and hyperparameter selection. If you reduce `EVAL_N`, record that change: it limits the notebook's local baseline, not automatically the full registered dataset consumed by managed evaluation.

The business risks differ. Incorrectly predicting `Entailment` can claim that a protection exists when the contract is silent. Missing a real contradiction can hide an exception. A correct verdict paired with an unrelated citation can still mislead a reviewer. Measure these dimensions separately before reducing them to a single score.

| Evaluation path        | Model input                                                     | Result source                                                    |
| ---------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| Local Bedrock baseline | System checklist and user contract through Converse             | Notebook score rows and `./eval_result` snapshots                |
| Managed custom scorer  | Registered test `query` and `response`, package plus base model | Pipeline S3 aggregates and per-record details                    |
| Managed LLM judge      | Generated answers plus task-specific rubrics                    | Bedrock evaluation scores and explanations under pipeline output |

The executable baseline identifier is `global.anthropic.claude-sonnet-4-6`; `BEDROCK_REGION` defaults to the SageMaker Region. Verify access before launching calls. The baseline uses `C.build_system(labels)` and `C.build_user(doc, no_think=False)`, whereas managed evaluation uses the single-string `query` ending with `/no_think`. These are different prompt presentations. Historical model names and benchmark tables elsewhere in the notebook are not evidence of your run.

## 1. Understand parsing and row construction

The local parser removes completed `<think>...</think>` blocks, looks inside a Markdown code fence if present, and scans braces to find a JSON object. The helper scorer has similar extraction logic and checks that the result is a dictionary. This is tolerant extraction, not strict JSON-only compliance: a reply with commentary can still be counted as parseable.

`score_contract` expands a prediction into one row per checklist key. Each row contains the gold/predicted label and sets of gold/predicted evidence identifiers. Missing entries remain wrong rather than disappearing from the denominator. Evidence strings that look like integers are converted into integers, and repeated citations collapse in a set.

The managed `score_record` helper also normalizes label case and whitespace, while the notebook's local row scorer compares the label directly. Record this distinction if you compare their results. Neither parser is a complete production schema validator. Separately check exact key membership, valid label values, integer evidence, in-document span bounds, and empty citations for `NotMentioned`.

## 2. Calculate the statistical metrics

Let \(N\) be the number of scored checklist decisions. Label accuracy is \(\sum_i \mathbf{1}[\hat y_i=y_i]/N\). For a particular class, precision is \(TP/(TP+FP)\), recall is \(TP/(TP+FN)\), and \(F_1=2PR/(P+R)\). The helpers return zero when the relevant denominator is zero.

For evidence, each eligible checklist item has a gold set \(G_i\) and a predicted set \(P_i\). Accumulate \(TP=\sum_i|G_i\cap P_i|\), \(FP=\sum_i|P_i-G_i|\), and \(FN=\sum_i|G_i-P_i|\), then compute F1. Only items with non-empty gold evidence participate in this calculation.

A worked arithmetic example, not a model result: with gold `{3, 4}` and prediction `{4, 9}`, there is one true-positive citation, one false positive, and one false negative. Precision and recall are each 0.5, so evidence F1 is 0.5. A correct label does not erase the wrong citation. In the implementation, evidence is scored independently of label correctness, so correct spans can also earn overlap credit beside a wrong verdict.

| Managed per-contract metric | Definition and limitation                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `label_correct`             | Fraction of reference keys with the correct normalized label                                              |
| `evidence_f1`               | F1 from citation counts pooled within this contract, excluding gold-empty items                           |
| `contradiction_correct`     | Fraction of gold `Contradiction` items correctly labeled; zero if the contract has no gold contradictions |
| `json_valid`                | Whether the tolerant parser extracts a JSON object; not a full schema check                               |
| `aggregate_reward_score`    | `0.6 * label_correct + 0.4 * evidence_f1`, the chosen workshop weighting                                  |

The local `contradiction_f1` is one-vs-rest F1 and penalizes false-positive contradictions. Managed `contradiction_correct` is recall-like on gold contradictions and is **not** that F1. Averaging its zero values for contracts with no contradictions also changes the interpretation. Report the metric's actual name and denominator.

The managed pipeline averages per-contract scores. The local evidence calculation instead pools all contracts' citation counts before calculating F1. In general, `mean(F1 per contract)` is not `F1(pooled counts)`. The local function reports percentages and the managed scorer reports fractions. Divide the local numbers by 100 before placing both on a 0-to-1 plot, and do not rank close scores as if the aggregation methods were identical.

Because gold-empty items are excluded from evidence F1, invented citations on those items are not penalized by this evidence metric. Extra checklist keys also do not enter the reference-key loop. Add schema and unsupported-citation checks when designing a production gate. A deterministic formula can still be incomplete or reward a shortcut.

## 3. Guard against reference leakage

Read [contractnli_scorer.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/contractnli_scorer.py). The managed pipeline uploads this self-contained file; it cannot import functions defined only in your notebook. Its `model_response` field contains generated text. The payload's `response` field can contain the gold answer inherited from the test dataset. Scoring `response` as the prediction would compare the reference with itself.

The notebook runs a paired sanity check before registering the evaluator:

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
    sagemaker_session=sess,
    role=role,
)
scorer_eval.hyperparameters.max_new_tokens = 8192
scorer_eval.hyperparameters.max_model_len = 24000
scorer_execution = scorer_eval.evaluate()
print(scorer_execution.arn)
```

`max_new_tokens` budgets generated output; `max_model_len` constrains the inference context. Neither changes the training sequence cap. Inspect the resolved temperature and other generation defaults printed by the notebook. A large output budget does not guarantee complete JSON if the input is too long or generation does not terminate as intended.

```python
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

The notebook saves local snapshots under `./eval_result`, including `frontier_metrics`, `frontier_rows`, `scorer_base`, `scorer_tuned`, and `summary`. These are convenient review artifacts; the managed copies remain in S3. Do not paste the notebook's hard-coded historical comparison printout into your own results table.

## 6. Add a semantic judge

Exact citation overlap cannot recognize two different clauses that state the same relevant restriction. The judge reads the clause text and asks whether it supports the verdict, supplementing rather than replacing the expert annotation comparison. A disagreement is a review candidate, not automatic proof that the annotation or model is wrong.

| Custom rubric             | Scale declared by the notebook | Review question                                                    |
| ------------------------- | ------------------------------ | ------------------------------------------------------------------ |
| `EvidenceSupportsVerdict` | 0 to 3                         | Do the cited clauses actually justify the committed verdicts?      |
| `CarveOutHandling`        | 0 to 3                         | Were exceptions and carve-outs interpreted appropriately?          |
| `ChecklistCompleteness`   | 0 or 1                         | Are all required entries and the requested JSON structure present? |

Read the full rubric instructions in the notebook before launching. They ask the judge to focus on final JSON and ignore reasoning preambles. Some wording in the carve-out rubric is ambiguous about silence versus exceptions; manually inspect explanations against the ContractNLI definitions rather than assuming the rubric is a formal legal test. Contract text and model output are untrusted data, including for a judge; instructions embedded inside them should not control grading.

::alert[The managed judge needs Bedrock service-role trust **and** identity permissions, including the applicable evaluation operations, model invocation, S3 access, and role passing. Adding `bedrock.amazonaws.com` to a trust policy does not grant `bedrock:CreateEvaluationJob`. Review [account setup](/01-prerequisites/2-account/) before starting another billed pipeline.]{type="warning"}

The notebook defines `custom_metrics`, serializes it with `json.dumps`, and passes the resulting **JSON string**, not a Python list:

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
    sagemaker_session=sess,
)
judge_execution = judge.evaluate()
print(judge_execution.arn)
```

After verifying pipeline success, use `judge_execution.show_results(limit=5, offset=0, show_explanations=True)`. The notebook reads per-record JSONL entries from `automatedEvaluationResult.scores`, with `metricName` and `result`. It treats returned scores as normalized values and preserves `null` results as missing rather than replacing them with zero. Inspect actual payloads to verify the scale before plotting.

Report scored count divided by total count for every rubric and model. A mean over one subset cannot be subtracted from a mean over a different subset and called a paired improvement. Where possible, compare the intersection of successfully judged contracts and report that selection alongside the full coverage. Keep judge model identity, rubric text, and execution ID with the result because model-based judgments can vary.

## Review and handoff

Inspect error directions, not only averages: `NotMentioned` to `Entailment`, missed contradictions, invalid citation IDs, incomplete checklists, and alternate-but-plausible evidence should be reviewed separately. Establish release thresholds for your own application before looking at the test results; this workshop supplies no guaranteed improvement, frontier parity, runtime, or cost target.

| Symptom                                | Next check                                                                             |
| -------------------------------------- | -------------------------------------------------------------------------------------- |
| Perfect scores for obviously poor text | Verify `model_response` versus reference fields with the paired sanity check           |
| Empty or old result files              | Confirm successful execution and exact S3 keys; reject the helper's cross-run fallback |
| Low parse rate                         | Inspect raw text, finish behavior, actual prompt, and generation budget                |
| High parse rate, poor evidence         | Review cited clauses and label errors rather than tuning formatting alone              |
| Judge access denied after inference    | Check the Bedrock evaluation identity policy in addition to trust                      |
| Unequal judge coverage                 | Report missingness and use a paired subset before comparing means                      |

Retain the package ARN, data versions, scorer version, generation configuration, execution IDs, and coverage with your findings. Continue to [Deployment](/02-serverless/04-deployment/) for an application-facing smoke test, or [Clean Up](/04-cleanup/) if you will not host the model.
