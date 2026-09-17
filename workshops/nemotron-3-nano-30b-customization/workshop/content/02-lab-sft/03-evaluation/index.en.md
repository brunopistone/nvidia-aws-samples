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

## Two Evaluation Methods

This lab uses both evaluation methods available in SageMaker AI managed evaluation.

**Statistical scoring** compares the predicted label and the cited span numbers against the expert annotation, using a custom scorer you write. It is exact, reproducible run to run, and costs nothing beyond the inference.

**LLM-as-a-Judge** has an evaluator model read the cited clause and rate whether it supports the verdict. It handles the case where a different clause states the same obligation, so the span numbers differ while the answer is right. With `evidence_f1` in the high sixties even after fine-tuning, a good part of what remains is citations a lawyer would accept and a string comparison will not.

Part 1 runs the scorer, Part 2 the judge.

Both are managed jobs, and between them they are the longest part of the lab. Budget **60-75 minutes** of waiting:

| Step | Duration | What dominates it |
| ------------------------ | ---------- | -------------------------------------------------------------- |
| Statistical scoring | 15-20 min | inference on 123 full-length contracts, per model |
| LLM-as-a-Judge | 45-55 min | the same inference, plus Nova Pro re-reading every response |

Neither needs supervision once launched. If you are short on time, run Part 1 and read Part 2's results here. The notebook can also load pre-computed results.

---

## Part 1: Statistical Scoring

::alert[Open the notebook **`lab-1-supervised-fine-tuning/3-evaluation.ipynb`**]

Each model returns one JSON object per contract with 17 entries. Scoring flattens that into one row per **(contract, checklist item)** pair: 123 contracts × 17 items = 2,091 rows on the full test set. Each row carries the gold label, the predicted label, the gold span set and the predicted span set.

Four metrics are computed from those rows:

| Metric | What it measures |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `accuracy` | Fraction of rows where the predicted label equals the gold label, after normalising case and whitespace. Citations do not enter it. |
| `contradiction_f1` | F1 for `Contradiction` as a one-vs-rest problem over the rows. It is the rare class, at roughly 10% of rows. |
| `evidence_f1` | Set overlap between cited and gold span numbers, micro-averaged over spans, restricted to rows where the gold cites at least one span. |
| `parse_failures` | Contracts whose output could not be parsed as JSON. Those rows still count, with a null prediction, so they depress accuracy. |

Details that matter when reading the numbers:

- **`evidence_f1` is micro over spans.** True positives, false positives and false negatives are summed across all rows *before* precision and recall are computed, so a row citing five spans influences the result five times as much as a row citing one.
- **Rows with no gold evidence are excluded from `evidence_f1`**, not scored as trivially correct. Otherwise the large share of rows whose gold answer is `NotMentioned`, which therefore cite nothing, would inflate the metric for free.
- **Spans are compared as integers.** There is no partial credit for an adjacent clause: citing span 47 when the gold is 48 scores the same as citing span 3.

### Which number to lead with

Not accuracy. A large share of this checklist is `NotMentioned`, so a model that answers `NotMentioned` to everything scores **43%** on `accuracy` without reading the contract at all, and exactly zero on evidence. Any accuracy figure has to be read against that floor.

`evidence_f1` has no such shortcut: span numbers are specific to the document in front of the model, so there is no distribution to learn. `contradiction_f1` is the second number to watch: it isolates whether the model can detect a clause that actively *conflicts* with a requirement, as opposed to one that is merely absent.

### The frontier baseline

Before measuring your own model, measure a model you did not train. Claude runs through the Bedrock Converse API and is scored by the same functions used everywhere else in the notebook:

```python
FRONTIER = "global.anthropic.claude-sonnet-5"

frontier_metrics, frontier_rows = evaluate_with_bedrock(EVAL_DOCS, FRONTIER)
```

`ask_bedrock`, `parse_verdicts`, `score_contract` and `metrics` are all defined in the notebook rather than imported, so you can read and change the scoring without leaving the page.

:::alert{header="Note" type="info"}
The Bedrock region is a named constant, `BEDROCK_REGION`, defaulting to `sess.boto_region_name`. It is kept separate from the SageMaker region because model availability differs by region. If a frontier model is not offered where you are running, point this at a region where it is.
:::

Prompting harder does not close the gap. Adding worked examples to the prompt makes the small base model **worse** at detecting contradictions, because demonstrations teach output shape and shape was never the bottleneck. Its problem is reading the contract, which no number of examples supplies. That is the argument for fine-tuning over prompt engineering on this task.

:::alert{header="About `Evaluator`" type="info"}
`sagemaker.ai_registry.evaluator.Evaluator` registers a custom scoring function in the SageMaker AI Registry. The `source` parameter points to a single Python file with a `lambda_handler` function. The pipeline uploads that file and executes it in an isolated container. `type=REWARD_FUNCTION` (from `sagemaker.ai_registry.air_constants`) is the evaluator type; the same interface is used for RL reward signals and evaluation scorers. Once registered, `Evaluator.get(name=...)` retrieves it by name so the same scorer can be reused across runs. [API documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-customization-evaluation-custom.html)
:::

### Register the custom scorer

`contractnli_scorer.py` implements the same comparison as a scorer the managed pipeline can run. It is registered in the AI Registry as an evaluator:

```python
from sagemaker.ai_registry.air_constants import REWARD_FUNCTION
from sagemaker.ai_registry.evaluator import Evaluator

SCORER_NAME = "contractnli-scorer"
try:
 scorer = Evaluator.get(name=SCORER_NAME)
 print("reusing registered scorer")
except Exception:
 scorer = Evaluator.create(name=SCORER_NAME, type=REWARD_FUNCTION,
 source="contractnli_scorer.py", role=role,
 sagemaker_session=sess, wait=True)
 scorer.refresh()
 print("registered scorer")
```

The `get`-then-`create` shape makes the cell re-runnable: registering the same name twice fails, and a restarted kernel should not force you to rename the evaluator.

A scorer is a single Python file with a `lambda_handler`. For each record it returns one object:

```python
{"id": "<record id>",
 "aggregate_reward_score": 0.83, # the headline number
 "metrics_list": [# as many named metrics as you like
 {"name": "label_correct", "value": 0.88, "type": "Reward"},
 {"name": "evidence_f1", "value": 0.76, "type": "Reward"},
 ]}
```

`type` is either `"Reward"` or `"Metric"`; both appear in the results. This scorer emits four: `label_correct`, `evidence_f1`, `contradiction_correct` and `json_valid`, deliberately the same four the notebook computes locally, so the two paths can be compared directly.

:::alert{header="Note" type="info"}
The file must be self-contained. The pipeline uploads it and executes it in its own container, where neither the notebook nor `contractnli.py` exists, so the scoring logic is duplicated on purpose. The notebook prints the scorer's `score_record` with `inspect.getsource` so you can read it rather than take it on trust.
:::

:::alert{header="Note" type="warning"}
Each record arrives with three fields that contain an answer: `model_response` (the model's output), `reference_answer` (the gold), and `response`, which is **also** the gold, carried over from the dataset's `response` column. The scorer reads `model_response`. Reading `response` instead would compare the gold answer against itself and return a perfect score on every record with no error raised, so the notebook sanity-checks the scorer with a deliberately poor answer before registering it.
:::

:::alert{header="Note" type="info"}
The evaluator type is `REWARD_FUNCTION`. That name comes from reinforcement learning, where the same interface supplies the training signal, but nothing stops you using it purely for evaluation, which is what we do here.
:::

### Run the evaluation

:::alert{header="About `CustomScorerEvaluator`" type="info"}
`sagemaker.train.evaluate.CustomScorerEvaluator` runs a registered scorer against a registered model package as a fully managed SageMaker pipeline, with no endpoint to provision and no infrastructure to configure. It calls the registered `Evaluator` for each record in the test dataset and attaches the results to the model package as lineage. The `evaluate_base_model=True` flag launches two parallel pipeline steps (base and fine-tuned) in the same job, so the before/after comparison is measured under identical compute conditions. [API documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-customization-evaluation.html)
:::

`CustomScorerEvaluator` runs inference and scoring for you, directly from the registered model package. Setting `evaluate_base_model=True` scores the **base and fine-tuned models in the same job**, with the same scorer, so the before/after comparison is measured under identical conditions:

```python
from sagemaker.train.evaluate import CustomScorerEvaluator

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
```

:::alert{header="Raise the generation limits, or the base model looks broken" type="warning"}
The two `hyperparameters` lines are not decoration. The prompt is a whole NDA, a median of about 2,800 tokens and up to 9,800 on this split, so the pipeline's default `max_model_len` will truncate the longest contracts, and its default `max_new_tokens` can cut a reply off before the JSON closes.

Either failure surfaces as an unparseable answer, which scores zero on all 17 items and is indistinguishable from a model that cannot read a contract. If your `json_valid` comes back low, check these two numbers before concluding anything about the model.
:::

No endpoint is involved. The model package is the artifact of record, so the evaluation attaches to it as lineage rather than living in a notebook. Nothing depends on an endpoint being up, which means no endpoint quota and nothing to tear down.

:::alert{header="Important" type="info"}
Endpoints (the next section) are for **serving** the model, not for measuring it. The base and fine-tuned NVIDIA are evaluated only by these pipelines, so the lab reports one set of numbers produced one way.
:::

### Read the results

The pipeline runs two parallel steps, `EvaluateBaseModel` and `EvaluateCustomModel`, so the step name in the S3 key tells you which model a results file belongs to:

```python
prefix = SCORER_OUTPUT.split(f"{BUCKET}/")[1]
base_scored = read_results(prefix, "EvaluateBaseModel")
tuned_scored = read_results(prefix, "EvaluateCustomModel")
```

Expect **15-20 minutes**. Almost all of it is inference: each prompt is a full NDA and there are 123 of them, per model. The two models are scored in parallel steps, so this is not double one model's time, and the scoring itself takes milliseconds.

### Statistical results

:::alert{header="Walk through the scorer metrics table" type="info"}
Show `scorer-metrics.png`. Point out all four metrics move in the right direction. Lead with the accuracy and evidence-F1 gains. Then call out `json_valid`: the base model returned parseable JSON on only 70% of contracts on this run, so its accuracy and evidence-F1 bars are partly format failure. The fine-tuned model has no such caveat.
:::

The scorer emits four metrics. Base and fine-tuned, scored in the same job on the full 123-contract test set (2,091 checklist decisions) with `max_epochs = 10`:

![Managed scorer metrics, base vs fine-tuned](/static/images/lab-1-sft/scorer-metrics.png)

| Metric | Base | Fine-tuned | Δ |
| ----------------------- | ---- | ---------- | ----- |
| `label_correct` | 51.8 | **84.2** | +32.4 |
| `evidence_f1` | 37.2 | **75.9** | +38.7 |
| `json_valid` | 70.0 | **~98** | ~+28 |

:::alert{header="The base model's json_valid was 70% on this run" type="warning"}
The base model returned parseable JSON on only 86 of 123 contracts (70%). An unparseable answer scores 0 on all 17 items, so the 51.8 accuracy and 37.2 evidence-F1 numbers above are part format failure. On the contracts it did answer, base is roughly 74% accuracy and 53% evidence-F1. The fine-tuned model does not have this problem: `json_valid` stays near 98%. The `/no_think` flag in the prompt suppresses the reasoning trace; without it the base model's output never closes the `<think>` block and never reaches the JSON.
:::

Adding the frontier baseline, which the notebook measures locally through Bedrock:

![Three models on accuracy and evidence-F1](/static/images/lab-1-sft/model-comparison.png)

| Model | Accuracy | Evidence-F1 | How |
| ------------------------------ | -------- | ----------- | -------------- |
| Base Nano 30B | 51.8 | 37.2 | managed scorer |
| Claude Sonnet 4.6 | 79.0 | 70.0 | local pass |
| **Fine-tuned Nano 30B (LoRA)** | **84.2** | **75.9** | managed scorer |

:::alert{header="Show the three-model comparison chart" type="info"}
Show `model-comparison.png` and then `statistical-scoring-chart.png`. The key message: a fine-tuned model with ~3B active parameters outperforms a frontier Claude model on this task, at roughly 8× lower inference cost per contract. Emphasize that the base model's low bars are partly a format failure (70% json_valid). The real story is how far fine-tuning moves things from a working starting point.
:::

![Statistical scoring chart](/static/images/lab-1-sft/statistical-scoring-chart.png)

**The fine-tuned model beats the frontier baseline on both metrics:** 84.2 vs 79.0 on accuracy, 75.9 vs 70.0 on evidence-F1. Unlike the base model comparison, this is a genuine reading gain: the fine-tuned model's `json_valid` is near 98%, so there are no formatting caveats on those numbers. The gain is also not subtle: 5 points on accuracy and nearly 6 on evidence-F1 against a Claude Sonnet model.

The gain against the starting point is even larger: **+32.4 accuracy and +38.7 evidence-F1** over the base, on a model you own and can serve yourself at a fraction of frontier API cost.

The notebook evaluates the whole held-out split by default so its local numbers and the managed pipeline's are computed on the same records. Lower `EVAL_N` for a faster pass and expect a couple of points of noise.

:::alert{header="Check json_valid before reading anything else, then move on" type="info"}
An answer that does not parse scores zero on all 17 of its contract's items, so `json_valid` is the denominator the other three metrics have to be read against. On this run it came back **70% base / ~98% fine-tuned**. The fine-tuned model's numbers are real measurements of contract comprehension. The base model's numbers are not: 30% of its answers never parsed, so accuracy 51.8 and evidence-F1 37.2 reflect both reading ability and format failure mixed together.

The base model is a reasoning model. Without `/no_think` in the prompt, it spends its whole generation budget inside `<think>` and never reaches the JSON. One line in the prompt is the difference between a model that looks completely broken and one that answers 70% of contracts. The notebook prints a caution automatically if `json_valid` ever drops below 95%, so if you see that warning, check the prompt records before drawing any conclusions about model quality.
:::

:::alert{header="Note" type="warning"}
The managed scorer runs **per record** and averages the results, so its `evidence_f1` is a macro average of per-contract F1, while the local pass computes corpus-level micro F1. They agree within about a point on this dataset, but they are different statistics. Do not present them as the same number. This is why the fine-tuned model's 1.8-point edge on evidence-F1 over Sonnet cannot be read as a ranking: the two rows are not computed identically. Label accuracy is unaffected: every contract has exactly 17 items, so a per-contract mean *is* the corpus figure.
:::

The pipeline also reports built-in ROUGE and BLEU alongside your metrics. Ignore them here: this model emits JSON, so string overlap with the reference stays high (BLEU ~98) no matter how many verdicts are wrong. That gap between "looks similar" and "is correct" is why a custom scorer was necessary.

### Behind the averages

Every number above is a mean over 123 contracts, and a mean hides the shape it came from. The pipeline also writes the per-record scores, so the notebook plots the distribution rather than taking the average on faith:

![Per-contract score distributions](/static/images/lab-1-sft/per-contract-spread.png)

:::alert{header="Explain the distribution plot" type="info"}
Show `per-contract-spread.png`. This is the most important chart for understanding *what kind* of improvement fine-tuning delivered. Walk through the three observations in the text below the image: the whole distribution shifted (not just a few contracts), citations spread more than labels, and contradiction_correct is granular because 10% frequency means few possible values per contract.
:::

Three things this shows that the table cannot:

- **The fine-tuned distribution barely overlaps the base.** On `label_correct`, the base median is around 0.75 with a wide spread; the fine-tuned box tightens around 0.85-0.88. That tight cluster is what a real improvement looks like: almost every contract got better, not a handful of lucky ones.
- **The citation gain is larger than the label gain.** `evidence_f1` moves from a wide base spread (median ~0.40) to a concentrated fine-tuned cluster (median ~0.78), the biggest jump of any metric. Some contracts still miss badly, so the average is a genuine mean of unlike cases.
- **`contradiction_correct` is the coarse metric.** With contradictions at about 10% of items, a contract can only score 0, 0.5 or 1. The averages are real but granular, and this is the metric to watch if you extend the lab or retrain.

`json_valid` tells the most important story: the base model has many contracts at 0 (failed to parse) while the fine-tuned model clusters tightly at 1.0. The base model's low accuracy and evidence-F1 are partly explained by those parse failures.

### How many epochs the metrics need

The four metrics do not converge at the same rate, and that is the practical lesson behind `max_epochs = 10`:

- **Output format is not what needs the epochs.** `json_valid` starts at 99.2 and ends at 98.4. There was nothing to learn here, which is exactly why the interesting movement is elsewhere.
- **Label accuracy flattens early.** Most of the +19 points is available within the first few epochs.
- **`evidence_f1` keeps climbing longest.** Citing the clause a reviewer would check is the hardest part of the task and the last to converge, which is why the lab does not stop at one or two epochs.

So if you shorten the run to save time, expect the citations to be what you give up first, and with them the parity against Sonnet, which the fine-tuned model reaches by less than a point.

:::alert{header="Your run will not match these decimals" type="info"}
Everything above is one measured run at `max_epochs = 10`. A fine-tune is not bit-reproducible, and on a 315-record training set the variance between runs is large enough to move `evidence_f1` by several points, enough to turn the parity with Sonnet into a narrow win or a narrow loss. Read the shape of the result (format fixed, accuracy and citations brought up to frontier level) rather than the third significant figure.
:::

This is also the run every other measured claim in the lab comes from: the cost-per-contract table in the [Lab 1 overview](../) and the token counts behind it. Those are serving-side measurements, so they depend on how long the model's output is rather than on how long it trained, but they were taken from this model.

---

## Part 2: LLM-as-a-Judge

:image[LLM-as-a-Judge concept]{src="/static/images/lab-1-sft/llm_judge.png" height=300}

:::alert{header="About `LLMAsJudgeEvaluator`" type="info"}
`sagemaker.train.evaluate.LLMAsJudgeEvaluator` runs an LLM-as-a-Judge evaluation pipeline powered by [Amazon Bedrock Evaluations](https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-judge.html). It accepts the same model package and dataset as `CustomScorerEvaluator`, plus a judge model (via `evaluator_model`) and a set of custom metrics defined as judge prompts (via `custom_metrics`). The evaluator model reads each model response and scores it against your metrics. [API documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-customization-evaluation-judge.html)
:::

`LLMAsJudgeEvaluator` takes the same model package and dataset, and runs Nova Pro as the judge:

```python
from sagemaker.train.evaluate import LLMAsJudgeEvaluator

judge = LLMAsJudgeEvaluator(
 model=model_package_arn,
 model_package_group=model_package_group_name,
 evaluator_model=EVALUATOR_MODEL,
 dataset=test_dataset,
 custom_metrics=custom_metrics_json, # a JSON string, not a list
 s3_output_path=JUDGE_OUTPUT,
 evaluate_base_model=True,
 sagemaker_session=sess,
)
```

Three custom metrics, each scoped to this task:

| Metric | Scale | Question it asks |
| ------------------------- | ----- | ----------------------------------------------------------------------------- |
| `EvidenceSupportsVerdict` | 0–3 | Do the cited clauses actually justify the verdict, whatever their span numbers? |
| `CarveOutHandling` | 0–3 | Were exceptions and carve-outs handled, rather than defaulted past? |
| `ChecklistCompleteness` | 0–1 | Valid JSON, an entry for all 17 keys, no commentary outside it. |

:::alert{header="Warning" type="warning"}
The judge prompts tell the evaluator to grade **only the final JSON object** and ignore the `<think>` block. These are reasoning models, and even with `/no_think` in the prompt the base model can emit a reasoning preamble the fine-tuned model does not. A judge shown all of it rewards confident-sounding reasoning that reaches the wrong label, and it rewards the more verbose model, which is the base one. This is the general lesson: a judge measures whatever its prompt lets it see, so check a judge metric's direction against ground truth before trusting it.
:::

:::alert{header="This is the slow step" type="warning"}
The judge run takes **45-55 minutes**, roughly three times the statistical job, and the longest single wait in the lab. Two models × 123 contracts are inferred first, then every response is read again by Nova Pro against three custom metrics, so the judging is a second full pass over the same volume of text. Launch it and come back; the notebook polls once a minute.
:::

### Judge results

Thirty held-out contracts, base and fine-tuned scored in the same job, full 30/30 coverage on all three metrics:

| Metric | Base | Fine-tuned | Δ | Coverage (base / tuned) |
| ------------------------- | ---- | ---------- | ----- | ----------------------------- |
| `ChecklistCompleteness` | 0.03 | 1.00 | +0.97 | 30/30 / 30/30 |
| `CarveOutHandling` | 0.52 | 0.69 | +0.17 | 30/30 / 30/30 |
| `EvidenceSupportsVerdict` | 0.19 | 0.81 | +0.62 | 30/30 / 30/30 |

Win rates: fine-tuned wins 93.3% (28/30), base wins 0%, ties 6.7%.

The notebook draws the same three metrics as grouped bars and as a radar, each annotated with the coverage figures from the table:

:::alert{header="Walk through these charts with participants" type="info"}
Show the bar chart first (`judge-comparison.png`). Lead with `ChecklistCompleteness`: base at 3% and fine-tuned at 100% is the single most vivid demonstration of what fine-tuning does to output format discipline. Then walk through `EvidenceSupportsVerdict` (+0.62) and `CarveOutHandling` (+0.17). Switch to the radar chart (`judge-radar.png`) to show the shape: the base model's triangle is tiny, the fine-tuned model fills most of the space. Ask participants which metric they'd use to gate a production deployment decision and why.
:::

![LLM-as-a-Judge, base vs fine-tuned](/static/images/lab-1-sft/judge-comparison.png)

![LLM-as-a-Judge radar](/static/images/lab-1-sft/judge-radar.png)

The radar is the quicker read of the two: the base model's triangle is very small, the fine-tuned model's covers most of the chart.

Read the three rows as three different *kinds* of result:

- **`ChecklistCompleteness` tells the format story.** Base at 0.03 vs fine-tuned at 1.00 is not a score gap, it is a categorical difference: the base model mostly fails to produce a complete 17-item checklist, while the fine-tuned model produces one on essentially every contract. This agrees with the 70% `json_valid` from Part 1. **Fine-tuning solved the output format problem completely.**
- **`EvidenceSupportsVerdict` (+0.62) shows the largest reading gain.** At 0.19 base vs 0.81 fine-tuned, the judge is finding that the fine-tuned model's cited clauses actually justify the verdict, not just happen to be nearby. This is the hardest thing to learn and the most meaningful improvement.
- **`CarveOutHandling` (+0.17) has the most room left.** At 0.69 the fine-tuned model is doing well but not great on reading exceptions. This is where more training data would pay off most.

All three point the same direction as the statistical scorer. Two completely different methods agree on what fine-tuning improved and by how much.

:::alert{header="Coverage can vary between runs" type="warning"}
This run achieved full 30/30 coverage on all three metrics, which made the deltas straightforward to read. Coverage is not guaranteed: the judge returns nothing when it cannot parse an answer, and which records it skips differs between models and between runs. Your chart will annotate each bar with the coverage count. If two models have different coverage on the same metric, the delta is not a before/after on the same contracts. Read it as directional, not exact.
:::

Judge scores also move a point or two between identical runs, because the judge is a model rather than a formula. Compare directions and the ordering of the deltas, not the third decimal place.

:::alert{header="Important: two separate IAM requirements" type="warning"}
The judge runs on Amazon Bedrock, so the SageMaker execution role needs **both** a trust relationship allowing `bedrock.amazonaws.com` to assume it, **and** an identity-based policy granting `bedrock:CreateEvaluationJob` (plus `GetEvaluationJob`, `ListEvaluationJobs`, `StopEvaluationJob`) and `bedrock:InvokeModel` on the evaluator model's inference profile.

The trust relationship alone is not enough, and `AmazonSageMakerFullAccess` does **not** grant `bedrock:CreateEvaluationJob`. If the second is missing, the inference steps succeed and the run fails roughly 25 minutes later at the metrics step with `AccessDeniedException ... not authorized to perform: bedrock:CreateEvaluationJob`. See [Bedrock evaluation permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/judge-service-roles.html).
:::

---

## Where the improvement comes from

The notebook closes with a failure analysis rather than a single score. It does not ask *how many* decisions were wrong (the table already covered that) but *which way* they were wrong. The chart is drawn for the frontier model, because that is the baseline a customer will compare against:

![claude-sonnet-5 error directions](/static/images/lab-1-sft/error-directions-sonnet.png)

:::alert{header="Discuss error directions with participants" type="info"}
This chart is the most important one in the lab for a deployment conversation. Point out that 37% of Claude Sonnet 4.6's errors go in the `NotMentioned → Entailment` direction: claiming a protection the contract never grants. Note also the large `NotMentioned → Contradiction` bar (26%): the model is often flagging a conflict that the contract never addresses. Ask participants: if you were shipping this to a legal team, which failure direction would you rather have? A false "yes, you're covered" or a missed item? The answer shapes what threshold you'd set before going to production.
:::

Claude Sonnet 4.6 got 434 of 2,091 decisions wrong on this run. `NotMentioned → Entailment` accounts for **161 of them (37%)**: the model reading a protection into a contract that never mentions it. But the second-largest error direction is `NotMentioned → Contradiction` (115, 26%): the model flagging a conflict that the contract never addresses. Together those two account for 63% of all errors. Add `Contradiction → Entailment` (33, 8%) and **over 70% of the errors involve a contract clause being misread rather than simply missed**. The remaining directions are the conservative failures: `NotMentioned → None` (parse failure, 45, 10%), `Entailment → Contradiction` (30, 7%), `Entailment → NotMentioned` (30, 7%).

That asymmetry matters more than the rate. A false "yes, you're protected" reaches a lawyer as a finished answer and is acted on; "missed one" is caught by the next reviewer. Two models at the same accuracy are not equally shippable if one of them fails in this direction, which is why a single number on a slide is not enough to make a deployment decision. Run the same plot for your own tuned model before you compare.

Worth running for the base model too: with `/no_think` in the prompt it returns parseable JSON on 99% of contracts, so its error directions are a real picture of how it reads a contract rather than an artefact of malformed output.

## Evaluate from the UI

Once your training job is complete, you can launch evaluation directly from the Studio UI.

From the training job details, click **Go to Custom Model**:

![Go to Custom Model](/static/images/lab-1-sft/studio-job-go-to-model.png)

In the custom model view, click **Evaluate** to launch the evaluation wizard:

![Custom Model Evaluate](/static/images/lab-1-sft/studio-custom-model-evaluate.png)

The evaluation setup allows you to:

- Choose evaluation type: **LLM-as-a-Judge**, **Custom Scorer**, or **Benchmarks**
- Select an evaluator model (Nova Pro, Claude, Mistral)
- Upload or select an existing test dataset
- Configure output location

![Evaluation Setup](/static/images/lab-1-sft/studio-evaluation-setup.png)

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach so the scorer and the judge prompts are version-controlled alongside the lab.
:::

:::alert{header="Short on time?" type="warning"}
If you don't want to wait for the managed scorer evaluation job to complete, the notebook supports downloading pre-computed evaluation results so you can skip ahead to comparing the base, frontier, and fine-tuned models.
:::
