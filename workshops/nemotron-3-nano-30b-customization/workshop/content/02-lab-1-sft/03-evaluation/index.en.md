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

**LLM-as-a-Judge** has an evaluator model read the cited clause and rate whether it supports the verdict. It handles the case where a different clause states the same obligation, so the span numbers differ while the answer is right — with `evidence_f1` in the high sixties even after fine-tuning, a good part of what remains is citations a lawyer would accept and a string comparison will not.

Part 1 runs the scorer, Part 2 the judge.

Both are managed jobs, and between them they are the longest part of the lab — budget **60-75 minutes** of waiting:

| Step                     | Duration   | What dominates it                                             |
| ------------------------ | ---------- | -------------------------------------------------------------- |
| Statistical scoring      | 15-20 min  | inference on 123 full-length contracts, per model              |
| LLM-as-a-Judge           | 45-55 min  | the same inference, plus Nova Pro re-reading every response     |

Neither needs supervision once launched. If you are short on time, run Part 1 and read Part 2's results here — the notebook can also load pre-computed results.

---

## Part 1: Statistical Scoring

::alert[📒 Open the notebook **`lab-1-supervised-fine-tuning/3-evaluation.ipynb`**]

Each model returns one JSON object per contract with 17 entries. Scoring flattens that into one row per **(contract, checklist item)** pair — 123 contracts × 17 items = 2,091 rows on the full test set. Each row carries the gold label, the predicted label, the gold span set and the predicted span set.

Four metrics are computed from those rows:

| Metric             | What it measures                                                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `accuracy`         | Fraction of rows where the predicted label equals the gold label, after normalising case and whitespace. Citations do not enter it.  |
| `contradiction_f1` | F1 for `Contradiction` as a one-vs-rest problem over the rows. It is the rare class, at roughly 10% of rows.                         |
| `evidence_f1`      | Set overlap between cited and gold span numbers, micro-averaged over spans, restricted to rows where the gold cites at least one span. |
| `parse_failures`   | Contracts whose output could not be parsed as JSON. Those rows still count, with a null prediction, so they depress accuracy.        |

Details that matter when reading the numbers:

- **`evidence_f1` is micro over spans.** True positives, false positives and false negatives are summed across all rows *before* precision and recall are computed, so a row citing five spans influences the result five times as much as a row citing one.
- **Rows with no gold evidence are excluded from `evidence_f1`**, not scored as trivially correct. Otherwise the large share of rows whose gold answer is `NotMentioned` — and which therefore cite nothing — would inflate the metric for free.
- **Spans are compared as integers.** There is no partial credit for an adjacent clause: citing span 47 when the gold is 48 scores the same as citing span 3.

### Which number to lead with

Not accuracy. A large share of this checklist is `NotMentioned`, so a model that answers `NotMentioned` to everything scores **43%** on `accuracy` without reading the contract at all — and exactly zero on evidence. Any accuracy figure has to be read against that floor.

`evidence_f1` has no such shortcut — span numbers are specific to the document in front of the model, so there is no distribution to learn. `contradiction_f1` is the second number to watch: it isolates whether the model can detect a clause that actively *conflicts* with a requirement, as opposed to one that is merely absent.

### The frontier baseline

Before measuring your own model, measure a model you did not train. Claude runs through the Bedrock Converse API and is scored by the same functions used everywhere else in the notebook:

```python
FRONTIER = "global.anthropic.claude-sonnet-5"

frontier_metrics, frontier_rows = evaluate_with_bedrock(EVAL_DOCS, FRONTIER)
```

`ask_bedrock`, `parse_verdicts`, `score_contract` and `metrics` are all defined in the notebook rather than imported, so you can read and change the scoring without leaving the page.

:::alert{header="Note" type="info"}
The Bedrock region is a named constant, `BEDROCK_REGION`, defaulting to `sess.boto_region_name`. It is kept separate from the SageMaker region because model availability differs by region — if a frontier model is not offered where you are running, point this at a region where it is.
:::

Prompting harder does not close the gap. Adding worked examples to the prompt makes the small base model **worse** at detecting contradictions, because demonstrations teach output shape and shape was never the bottleneck — its problem is reading the contract, which no number of examples supplies. That is the argument for fine-tuning over prompt engineering on this task.

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
 "aggregate_reward_score": 0.83,          # the headline number
 "metrics_list": [                         # as many named metrics as you like
     {"name": "label_correct", "value": 0.88, "type": "Reward"},
     {"name": "evidence_f1",   "value": 0.76, "type": "Reward"},
 ]}
```

`type` is either `"Reward"` or `"Metric"`; both appear in the results. This scorer emits four — `label_correct`, `evidence_f1`, `contradiction_correct` and `json_valid` — deliberately the same four the notebook computes locally, so the two paths can be compared directly.

:::alert{header="Note" type="info"}
The file must be self-contained. The pipeline uploads it and executes it in its own container, where neither the notebook nor `contractnli.py` exists, so the scoring logic is duplicated on purpose. The notebook prints the scorer's `score_record` with `inspect.getsource` so you can read it rather than take it on trust.
:::

:::alert{header="Note" type="warning"}
Each record arrives with three fields that contain an answer: `model_response` (the model's output), `reference_answer` (the gold), and `response` — which is **also** the gold, carried over from the dataset's `response` column. The scorer reads `model_response`. Reading `response` instead would compare the gold answer against itself and return a perfect score on every record with no error raised, so the notebook sanity-checks the scorer with a deliberately poor answer before registering it.
:::

:::alert{header="Note" type="info"}
The evaluator type is `REWARD_FUNCTION`. That name comes from reinforcement learning, where the same interface supplies the training signal — but nothing stops you using it purely for evaluation, which is what we do here.
:::

### Run the evaluation

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
The two `hyperparameters` lines are not decoration. The prompt is a whole NDA — a median of about 2,800 tokens and up to 9,800 on this split — so the pipeline's default `max_model_len` will truncate the longest contracts, and its default `max_new_tokens` can cut a reply off before the JSON closes.

Either failure surfaces as an unparseable answer, which scores zero on all 17 items and is indistinguishable from a model that cannot read a contract. If your `json_valid` comes back low, check these two numbers before concluding anything about the model.
:::

No endpoint is involved. The model package is the artifact of record, so the evaluation attaches to it as lineage rather than living in a notebook — and nothing depends on an endpoint being up, which means no endpoint quota and nothing to tear down.

:::alert{header="Important" type="info"}
Endpoints (the next section) are for **serving** the model, not for measuring it. The base and fine-tuned Qwen are evaluated only by these pipelines, so the lab reports one set of numbers produced one way.
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

The scorer emits four metrics. Base and fine-tuned, scored in the same job on the full 123-contract test set — 2,091 checklist decisions — with `max_epochs = 10`:

![Managed scorer metrics, base vs fine-tuned](/static/images/lab-1-sft/scorer-metrics.png)

| Metric                  | Base | Fine-tuned | Δ     |
| ----------------------- | ---- | ---------- | ----- |
| `label_correct`         | 64.8 | **83.8**   | +19.0 |
| `evidence_f1`           | 48.9 | **68.9**   | +20.0 |
| `contradiction_correct` | 37.9 | **64.4**   | +26.5 |
| `json_valid`            | 99.2 | **98.4**   | −0.8  |

Adding the frontier baseline, which the notebook measures locally through Bedrock:

![Three models on accuracy and evidence-F1](/static/images/lab-1-sft/model-comparison.png)

| Model                          | Accuracy | Evidence-F1 | How            |
| ------------------------------ | -------- | ----------- | -------------- |
| Base Qwen3 4B                  | 64.8     | 48.9        | managed scorer |
| Claude Sonnet 5                | 83.6     | 67.1        | local pass     |
| **Fine-tuned Qwen3 4B (LoRA)** | **83.8** | **68.9**    | managed scorer |

![Statistical scoring chart](/static/images/lab-1-sft/statistical-scoring-chart.png)

**A 4B model fine-tuned on 315 contracts reaches Claude Sonnet 5's level** — 83.8 against 83.6 on accuracy, 68.9 against 67.1 on evidence-F1. It is nominally ahead on both, but do not sell it as a win: 0.2 points is far inside run-to-run noise, and the two rows are not computed by the same averaging (see the note below). Parity is the claim the data supports, and it is the one worth having — the same quality at roughly **8x lower cost per contract**, on a model you own and can serve yourself.

The gain to quote is the one against the starting point: **+19 accuracy, +20 evidence-F1, +26.5 on contradiction detection**, over the same base weights on the same records.

The notebook evaluates the whole held-out split by default so its local numbers and the managed pipeline's are computed on the same records. Lower `EVAL_N` for a faster pass and expect a couple of points of noise.

:::alert{header="Check json_valid before reading anything else — then move on" type="info"}
An answer that does not parse scores zero on all 17 of its contract's items, so `json_valid` is the denominator the other three metrics have to be read against. Here it comes back **99.2% base / 98.4% fine-tuned**, which is the useful case to see: both columns are real measurements of contract comprehension, not formatting artefacts, so the +19 / +20 gain is a *reading* gain.

That is worth dwelling on, because it did not come free. The base model is a reasoning model, and without `/no_think` in the prompt it spends its whole generation budget inside `<think>` and never reaches the JSON — which reads as a broken model when it is really a prompting bug. One line in the prompt is the difference between a base model that looks unusable and one that scores a respectable 64.8. The notebook prints a caution automatically if `json_valid` ever drops below 95%.

Note the fine-tuned model is 0.8 points *lower* — one single contract out of 123. That is noise, not a regression.
:::

:::alert{header="Note" type="warning"}
The managed scorer runs **per record** and averages the results, so its `evidence_f1` is a macro average of per-contract F1, while the local pass computes corpus-level micro F1. They agree within about a point on this dataset, but they are different statistics — do not present them as the same number. This is why the fine-tuned model's 1.8-point edge on evidence-F1 over Sonnet cannot be read as a ranking: the two rows are not computed identically. Label accuracy is unaffected: every contract has exactly 17 items, so a per-contract mean *is* the corpus figure.
:::

The pipeline also reports built-in ROUGE and BLEU alongside your metrics. Ignore them here: this model emits JSON, so string overlap with the reference stays high (BLEU ~98) no matter how many verdicts are wrong. That gap between "looks similar" and "is correct" is why a custom scorer was necessary.

### Behind the averages

Every number above is a mean over 123 contracts, and a mean hides the shape it came from. The pipeline also writes the per-record scores, so the notebook plots the distribution rather than taking the average on faith:

![Per-contract score distributions](/static/images/lab-1-sft/per-contract-spread.png)

Three things this shows that the table cannot:

- **The gain is a shift of the whole distribution, not a few rescued contracts.** On `label_correct` the base box runs about 0.59-0.77 around a median of 0.65; the fine-tuned box is a tight 0.83-0.88 around 0.85. The two barely overlap, which is what a real improvement looks like — every contract got better, not a handful.
- **The citations spread out much more than the labels.** `evidence_f1` moves from roughly 0.40-0.58 (median 0.50) to 0.61-0.82 (median 0.73), a wider box on both sides. Some contracts cite almost perfectly and some still miss badly, so the 68.9 average is a genuine mean of unlike cases rather than a typical contract's score.
- **`contradiction_correct` is the coarse metric.** The base median is 0.33 with a box from 0 to 0.5; the tuned median is 0.67 with a box from 0.5 to 1.0. Contradictions are about 10% of items, so a contract with one or two of them can only score 0, 0.5 or 1 — the averages are real but granular, and this is the metric to watch if you extend the lab.

`json_valid` is the flat one: both models sit on 1.0 with a single zero outlier each. One contract failed to parse per model, which is the whole of the 0.8-point difference in the table.

### How many epochs the metrics need

The four metrics do not converge at the same rate, and that is the practical lesson behind `max_epochs = 10`:

- **Output format is not what needs the epochs.** `json_valid` starts at 99.2 and ends at 98.4 — there was nothing to learn here, which is exactly why the interesting movement is elsewhere.
- **Label accuracy flattens early.** Most of the +19 points is available within the first few epochs.
- **`evidence_f1` keeps climbing longest.** Citing the clause a reviewer would check is the hardest part of the task and the last to converge, which is why the lab does not stop at one or two epochs.

So if you shorten the run to save time, expect the citations to be what you give up first — and with them the parity against Sonnet, which the fine-tuned model reaches by less than a point.

:::alert{header="Your run will not match these decimals" type="info"}
Everything above is one measured run at `max_epochs = 10`. A fine-tune is not bit-reproducible, and on a 315-record training set the variance between runs is large enough to move `evidence_f1` by several points — enough to turn the parity with Sonnet into a narrow win or a narrow loss. Read the shape of the result (format fixed, accuracy and citations brought up to frontier level) rather than the third significant figure.
:::

This is also the run every other measured claim in the lab comes from: the cost-per-contract table in the [Lab 1 overview](../) and the token counts behind it. Those are serving-side measurements, so they depend on how long the model's output is rather than on how long it trained — but they were taken from this model.

---

## Part 2: LLM-as-a-Judge

`LLMAsJudgeEvaluator` takes the same model package and dataset, and runs Nova Pro as the judge:

```python
from sagemaker.train.evaluate import LLMAsJudgeEvaluator

judge = LLMAsJudgeEvaluator(
    model=model_package_arn,
    model_package_group=model_package_group_name,
    evaluator_model=EVALUATOR_MODEL,
    dataset=test_dataset,
    custom_metrics=custom_metrics_json,   # a JSON string, not a list
    s3_output_path=JUDGE_OUTPUT,
    evaluate_base_model=True,
    sagemaker_session=sess,
)
```

Three custom metrics, each scoped to this task:

| Metric                    | Scale | Question it asks                                                              |
| ------------------------- | ----- | ----------------------------------------------------------------------------- |
| `EvidenceSupportsVerdict` | 0–3   | Do the cited clauses actually justify the verdict, whatever their span numbers? |
| `CarveOutHandling`        | 0–3   | Were exceptions and carve-outs handled, rather than defaulted past?           |
| `ChecklistCompleteness`   | 0–1   | Valid JSON, an entry for all 17 keys, no commentary outside it.                |

:::alert{header="Warning" type="warning"}
The judge prompts tell the evaluator to grade **only the final JSON object** and ignore the `<think>` block. These are reasoning models, and even with `/no_think` in the prompt the base model can emit a reasoning preamble the fine-tuned model does not. A judge shown all of it rewards confident-sounding reasoning that reaches the wrong label — and it rewards the more verbose model, which is the base one. This is the general lesson: a judge measures whatever its prompt lets it see, so check a judge metric's direction against ground truth before trusting it.
:::

:::alert{header="This is the slow step" type="warning"}
The judge run takes **45-55 minutes** — roughly three times the statistical job, and the longest single wait in the lab. Two models × 123 contracts are inferred first, then every response is read again by Nova Pro against three custom metrics, so the judging is a second full pass over the same volume of text. Launch it and come back; the notebook polls once a minute.
:::

### Judge results

All 123 held-out contracts, base and fine-tuned scored in the same job:

| Metric                    | Base | Fine-tuned | Δ     | Judge coverage (base / tuned) |
| ------------------------- | ---- | ---------- | ----- | ----------------------------- |
| `ChecklistCompleteness`   | 0.98 | 0.98       | ~0    | 122/123 / 123/123             |
| `CarveOutHandling`        | 0.46 | 0.59       | +0.13 | 123/123 / 123/123             |
| `EvidenceSupportsVerdict` | 0.31 | 0.45       | +0.14 | **25/123 / 53/123**           |

The notebook draws the same three metrics as grouped bars and as a radar, each annotated with the coverage figures from the table:

![LLM-as-a-Judge, base vs fine-tuned](/static/images/lab-1-sft/judge-comparison.png)

![LLM-as-a-Judge radar](/static/images/lab-1-sft/judge-radar.png)

The radar is the quicker read of the two: a metric that regressed shows as a dent in the shape rather than as two bars you have to compare by eye.

Read the three rows as three different *kinds* of result, because that is what tells you what more training can and cannot fix:

- **`ChecklistCompleteness` is already solved and stays solved.** Both models score 0.98 — all 17 items in valid JSON, essentially every time — so there is nothing here for fine-tuning to win. This agrees with `json_valid` at 99.2 / 98.4 from Part 1: output shape was never the bottleneck on this task. **A judge metric that both models max out has stopped discriminating**, and should be read as a pass/fail gate rather than a score.
- **`CarveOutHandling` (+0.13) is the honest, comparable gain.** Full coverage on both sides, the same 123 contracts, so the delta is a genuine before/after. On the 0–3 scale the judge was given, that is roughly 1.4 to 1.8 out of 3 — real movement, and still plenty of headroom. Reading an exception correctly is the hardest thing on the checklist, and 423 contracts is not enough supervision to close it. **More data will help here; more epochs will help less.** This is the judge number worth quoting.
- **`EvidenceSupportsVerdict` (+0.14) moves the most and is the one delta you should not quote.** It tracks `evidence_f1` from the statistical pass, but coverage is 25 and 53 of 123 — see the note below.

Both surviving deltas point the same way as the statistical scoring, which is the useful part: two methods with nothing in common agree on the direction and roughly on the ranking of what improved.

:::alert{header="Check the judge's coverage before reading a delta" type="warning"}
The chart annotates each metric with how many records the judge actually scored, as `base scored N/123 / tuned scored N/123`. On this run `CarveOutHandling` came back complete for both models, `ChecklistCompleteness` missed a single base record, and `EvidenceSupportsVerdict` came back on only **25/123 for the base model and 53/123 for the tuned one** — the judge returns nothing when it cannot decide, and it declines on different records for each model.

When coverage is well below 123 and unequal between the two, the "delta" is a comparison of two different subsets of contracts, not a before/after on the same ones. Read it as directional at best, and check the annotation on your own chart before quoting any of these three numbers — which metric loses coverage is not fixed between runs.
:::

Judge scores also move a point or two between identical runs, because the judge is a model rather than a formula. Compare directions and the ordering of the deltas, not the third decimal place.

:::alert{header="Important — two separate IAM requirements" type="warning"}
The judge runs on Amazon Bedrock, so the SageMaker execution role needs **both** a trust relationship allowing `bedrock.amazonaws.com` to assume it, **and** an identity-based policy granting `bedrock:CreateEvaluationJob` (plus `GetEvaluationJob`, `ListEvaluationJobs`, `StopEvaluationJob`) and `bedrock:InvokeModel` on the evaluator model's inference profile.

The trust relationship alone is not enough, and `AmazonSageMakerFullAccess` does **not** grant `bedrock:CreateEvaluationJob`. If the second is missing, the inference steps succeed and the run fails roughly 25 minutes later at the metrics step with `AccessDeniedException ... not authorized to perform: bedrock:CreateEvaluationJob`. See [Bedrock evaluation permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/judge-service-roles.html).
:::

---

## Where the improvement comes from

The notebook closes with a failure analysis rather than a single score. It does not ask *how many* decisions were wrong — the table already said that — but *which way* they were wrong. The chart is drawn for the frontier model, because that is the baseline a customer will compare against:

![claude-sonnet-5 error directions](/static/images/lab-1-sft/error-directions-sonnet.png)

Claude Sonnet 5 got 342 of 2,091 decisions wrong, and they are not spread evenly across the six possible confusions. `NotMentioned → Entailment` alone accounts for **164 of them, 48%** — the model reading a protection into a contract that never mentions it. Add `Contradiction → Entailment` (43, 13%) and **roughly three fifths of its errors point the same way**: towards claiming the reviewer is covered when the document does not say so. The other four directions are the conservative failures — `NotMentioned → Contradiction` (46, 13%), `Entailment → NotMentioned` (38, 11%), `Contradiction → NotMentioned` (36, 10%), `Entailment → Contradiction` (15, 4%).

That asymmetry matters more than the rate. A false "yes, you're protected" reaches a lawyer as a finished answer and is acted on; "missed one" is caught by the next reviewer. Two models at the same accuracy are not equally shippable if one of them fails in this direction, which is why a single number on a slide is not enough to make a deployment decision — run the same plot for your own tuned model before you compare.

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
