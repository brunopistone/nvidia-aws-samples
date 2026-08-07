---
title: "Data Preparation"
weight: 1
---

## About the Dataset

**[ContractNLI](https://stanfordnlp.github.io/contract-nli/)** (Koreeda & Manning, *Findings of EMNLP 2021*) is a document-level natural language inference dataset for contracts. It contains 607 real non-disclosure agreements collected from EDGAR filings and the public web, each annotated against a fixed 17-point checklist by legal experts.

For every (contract, checklist item) pair the annotation gives:

| Field    | Meaning                                                       |
| -------- | ------------------------------------------------------------- |
| `choice` | `Entailment` / `Contradiction` / `NotMentioned`                |
| `spans`  | the span numbers a lawyer would point to as justification     |

The splits are **document-level** and stratified by source format, so no contract appears in two splits: 423 train / 61 dev / 123 test. The dataset is released under **CC-BY-4.0**, and the licence file ships with the download.

:::alert{header="Note" type="info"}
The annotations are human expert labels, not model-generated, and the primary metrics are scored programmatically against them. That combination is what makes the headline numbers in this lab trustworthy. A judge model *is* used, but only in Part 2 of [Evaluation](../03-evaluation) and only to explain the residual the deterministic scorer cannot — never to produce the numbers the lab reports.
:::

---

## Option 1: Prepare Data with Code

::alert[📒 Open the notebook **`lab-1-supervised-fine-tuning/1-prepare-data.ipynb`**]

Select `Python 3 (ipykernel)` for the notebook kernel.

:image[Kernel Selection]{src="/static/images/lab-1-sft/kernel-selection.png" width=400 height=300}

### What `contractnli.py` gives you

Every notebook in the lab opens with `import contractnli as C`. The module holds only the unglamorous parts — fetching the data, reading it, and rendering the prompt. Anything that is the *lesson* (calling a model, scoring an answer) stays in the notebooks. This is its whole surface, and the notebook prints a version of this table so you can read it in place:

| Call                         | Returns                                  | Used for                                     |
| ---------------------------- | ---------------------------------------- | -------------------------------------------- |
| `C.ensure_dataset("./data")` | the unpacked path                        | downloads the ContractNLI archive once       |
| `C.load(split)`              | `(documents, checklist)`                 | reads `train`, `dev` or `test`               |
| `C.doc_spans(doc)`           | `[(0, "text"), (1, "text"), ...]`        | one contract as numbered clauses             |
| `C.gold_for(doc)`            | `{"nda-1": {"choice", "spans"}, ...}`    | the expert answer for one contract           |
| `C.build_prompt(doc, labels)` | one string                              | **the whole request** — instruction, contract, checklist |

The module also carries a two-turn `messages` variant of the same instruction — `C.build_system(labels)`, `C.build_user(doc)` and `C.build_messages(doc, labels, completion=None)`. Nothing in data preparation uses it. It exists for the callers that need role-tagged turns rather than a string: the frontier baseline in [Evaluation](../03-evaluation), which goes through the Bedrock Converse API, and the serving checks in [Deployment](../04-deployment).

:::alert{header="The two formats are not interchangeable" type="warning"}
Both carry the same instruction, the same checklist and the same `/no_think` switch, but they order them differently: `build_prompt` puts the contract *before* the checklist, while `build_messages` puts the checklist in the system turn, *ahead* of the contract. So they are not the same string, and neither is a drop-in for the other.

Training uses `build_prompt` and only `build_prompt`. If you reword the instruction, change it in both places or in neither.
:::

### Dataset structure

The notebook downloads the dataset once and loads the document-level splits:

```python
import contractnli as C

C.ensure_dataset("./data")
train_docs, labels = C.load("train")
dev_docs, _ = C.load("dev")
test_docs, _ = C.load("test")
```

Each document is a plain dict carrying the contract text, the span offsets, and the expert annotation set. `labels` is the checklist itself — 17 hypotheses with short descriptions. The notebook prints the fields of a real document and then demonstrates each helper on it, so you can see `doc['spans']` become numbered clauses and `doc['annotation_sets']` become one gold entry per checklist item.

### The prompt: one string

The prompt is the single most important design decision in the lab. It is defined **once**, in `contractnli.py`, and used by every notebook — data preparation here, the frontier baseline during evaluation, and the serving checks after deployment. A second copy pasted into a notebook is how train/serve skew gets introduced.

Everything the model ever sees is one string, with three slots filled in: the number of checklist items, the contract as numbered spans, and the checklist itself. Print it to see exactly what the model will receive:

```python
print(C.build_prompt(train_docs[0], labels))
```

Rendered, and truncated:

```
You are a contract review assistant. You review a non-disclosure agreement (NDA)
against a fixed checklist of 17 legal hypotheses.

For EACH hypothesis, decide:
- "Entailment": the contract states or implies the hypothesis is true.
- "Contradiction": the contract states something that conflicts with the hypothesis.
- "NotMentioned": the contract does not address it.

Also cite the span numbers that justify the decision ... Read exceptions and
carve-outs carefully: a clause with an exception may contradict a hypothesis
stated absolutely.

CONTRACT (numbered spans):
[0] NAVIDEC, INCORPORATED
[1] TRADE SECRET/NON-DISCLOSURE AGREEMENT
[3] ... shall not disclose or cause to be disclosed ... to any person, entity,
    business or other individual or company without the prior written permission ...
    ... (remaining spans)

CHECKLIST:
nda-5: Receiving Party may share some Confidential Information with some of
       Receiving Party's employees. (Sharing with employees)
    ... (17 items total, always in the same order)

Respond with JSON only, no other text:
{"nda-1": {"label": "Entailment|Contradiction|NotMentioned", "evidence": [span numbers]}, ...}
Include an entry for every hypothesis key listed above.

/no_think
```

Note the order: the checklist comes *after* the contract, so the last thing the model reads before answering is what it is being asked. Everything except the contract is a fixed 3,234 characters — of which the checklist block alone is 2,324 — identical in every record.

Prompt length, counted with the Qwen3 4B tokeniser over the training split: median **2,870 tokens**, mean **3,116**, longest **14,016**. The whole contract goes in every time, because any of the 17 items could be decided by any clause.

### Format your dataset for training and validation

One training record is **one contract with all 17 verdicts** in the completion — 423 records, each carrying 17 supervised decisions plus evidence spans, so about 7,200 labelled judgements.

Serverless customization accepts a record as a `prompt`/`completion` pair, and for a single-turn task that is the simpler of the two shapes it takes: everything the model reads goes in `prompt`, the one thing it must produce goes in `completion`, and there are no roles or turn boundaries to get right.

```python
label_keys = list(labels.keys())          # the checklist's own order


def gold_json(doc):
    """The expert answer for one contract, as the exact JSON the model must emit."""
    g = C.gold_for(doc)
    return json.dumps({k: {"label": g[k]["choice"], "evidence": list(g[k]["spans"])}
                       for k in label_keys if k in g})


def make_records(docs):
    """Training records: one prompt/completion pair per contract."""
    return [{"prompt": C.build_prompt(d, labels), "completion": gold_json(d)}
            for d in docs]
```

`gold_json` only renames the dataset's fields — `choice` becomes `label`, `spans` becomes `evidence`. The judgement is the annotator's, not ours. Example formatted row (contents truncated):

```json
{"prompt":     "You are a contract review assistant... CONTRACT (numbered spans):\n[0] ... CHECKLIST: ... /no_think",
 "completion": "{\"nda-11\": {\"label\": \"NotMentioned\", \"evidence\": []}, ...}"}
```

The recipes accept this alongside the role-tagged `messages` shape. There is no field mapping to configure.

Three deliberate choices:

- **Only the completion is supervised.** The recipe masks the loss on the prompt (`ignore_prompt_labels: true`), so the model learns to *produce* the JSON rather than to reproduce the contract. The field split is what tells the trainer where the loss starts — the boundary between context and answer is the `prompt`/`completion` boundary itself.
- **The prompt at training time is the prompt at inference time.** Both come from `C.build_prompt`. The notebook asserts this for all three splits rather than trusting it.
- **We keep the dataset's own document-level splits.** A random row-level split would put spans from the same contract in both train and test, and the reported numbers would be meaningless.

:::alert{header="Note" type="warning"}
Qwen3 4B caps a record at `dataset_max_len = 4096` tokens and **drops** — does not truncate — anything longer, so only **315 of the 423** contracts actually train at the default. The training log says so: `Final trainset size: 315, val dataset size: 41`. Raise `dataset_max_len` in the next section if you want all of them; it accepts up to 131072.
:::

### Format your dataset for testing

The test split keeps `query`/`response` instead of `prompt`/`completion`, because the evaluation pipeline's `CustomScorerEvaluator` pins the task to `gen_qa` — a format whose fields are exactly `query`, `response` and an optional `system`. It cannot read a `messages` array.

The same two strings therefore go out under genqa's names, which makes the test split a renaming rather than a second prompt. Because the whole instruction already lives in the prompt, the optional `system` field is simply not used:

```python
def make_test_records(docs):
    return [{"query": C.build_prompt(d, labels), "response": gold_json(d)}
            for d in docs]


records = {"train": make_records(train_docs),
           "val":   make_records(dev_docs),
           "test":  make_test_records(test_docs)}
```

That is the advantage this shape has here: `query` is the training prompt renamed, so there is nothing to keep in sync between the two formats, and no dependency on the evaluation container passing a `system` field through correctly. The notebook asserts it for every split — each stored prompt must equal `C.build_prompt(doc, labels)` for its document, and each completion must parse as non-empty JSON.

### Write to disk and upload to Amazon S3

Each split is written as `./sft_data/<split>/dataset.jsonl` — one JSON object per line, the shape `DataSet.create` validates before registering it. The directory names are the keys of `records`, so the `dev` documents are written as **`val`**, which is the name notebook 2 fetches the validation set by. The train file comes to 6.8 MB over its 423 lines, about 16 KB a record.

```python
from config import DATASET_PREFIX

local = pathlib.Path("./sft_data")
if local.exists():
    shutil.rmtree(local)          # a re-run cannot leave a stale file behind

for name, rows in records.items():
    d = local / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "dataset.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

input_path = (f"{default_prefix}/datasets/{DATASET_PREFIX}" if default_prefix
              else f"datasets/{DATASET_PREFIX}")

s3_paths = {}
for name in records:
    key = f"{input_path}/{name}/dataset.jsonl"
    s3_client.upload_file(str(local / name / "dataset.jsonl"), bucket_name, key)
    s3_paths[name] = f"s3://{bucket_name}/{key}"
```

The key joins `DATASET_PREFIX` (`contractnli-nda-review`, from `config.py`) to `default_prefix`, which the setup cell read from `sess.default_bucket_prefix` — a leading key segment when a SageMaker config sets one, and `None` otherwise, hence the conditional. Each file lands at `s3://<bucket>/[<prefix>/]datasets/contractnli-nda-review/<split>/dataset.jsonl`.

### Create SageMaker AI Datasets

The key step for serverless customization is registering those objects as **SageMaker AI Datasets**. What that buys you is a *name*: notebook 2 calls `DataSet.get(name=f"{DATASET_PREFIX}-train")` and hands the result to `SFTTrainer(training_dataset=...)`, which resolves it to the entry's name and version rather than an S3 URI — so the job records which dataset it consumed.

```python
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique


def register(name, source, technique=None):
    kwargs = dict(name=name, source=source, wait=True)
    if technique is not None:
        kwargs["customization_technique"] = technique
    return DataSet.create(**kwargs)


training_dataset = register(f"{DATASET_PREFIX}-train", s3_paths["train"], CustomizationTechnique.SFT)
val_dataset = register(f"{DATASET_PREFIX}-val", s3_paths["val"], CustomizationTechnique.SFT)
test_dataset = register(f"{DATASET_PREFIX}-test", s3_paths["test"])
```

| dataset                        | technique | consumed by                          |
| ------------------------------ | --------- | ------------------------------------ |
| `contractnli-nda-review-train` | `SFT`     | notebook 2, `training_dataset=`      |
| `contractnli-nda-review-val`   | `SFT`     | notebook 2, `validation_dataset=`    |
| `contractnli-nda-review-test`  | none      | notebook 3, `dataset=`               |

`wait=True` blocks until each import reaches `Available`, and raises if it reaches `ImportFailed` instead.

The technique is stored as the search keyword `customization_technique:sft` — a label to find the dataset by, not a constraint. Nothing reads it back at training time, since the trainer knows its own technique. The enum offers `SFT`, `DPO` and `RLVR` and nothing for evaluation data, which is why the test split is registered without one.

`create` also downloads each file and matches it against the formats the registry knows — the prompt/completion shape for train and val, `genqa` for the test split — so the wrong shape fails here rather than inside a training job. It reads only the file's **first record**, which is why the build cell's per-record assertions are the stronger guarantee. The match is pass/fail and no format name is stored in the entry, so the registry console shows an empty **Format** column for every dataset. That is not a sign anything is wrong with yours.

:::alert{header="Re-registering does not re-read your data" type="warning"}
`create` overwrites nothing — it reads the current version and imports the next major one, so a second run leaves `2.0.0` beside `1.0.0`, and a lookup by bare name returns the latest.

But the entry records only a bucket and a key, with **no version id or checksum**, and the upload cell always writes the same key. Every version therefore resolves to the *same* `dataset.jsonl`: the version list is a history of registrations, not of your data. Re-upload before you re-register, and never expect `1.0.0` to still hold the records it was created with.
:::

## Option 2: Create Datasets with UI

If you have an already formatted dataset in JSONL format, you can also create datasets directly from the SageMaker Studio UI. Navigate to **Assets** → **Datasets** and click **Upload Dataset**:

![Studio Datasets Upload](/static/images/lab-1-sft/studio-datasets-upload.png)

The UI shows the required data format for each customization technique and allows you to upload JSONL files or specify an S3 URI.

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach for more flexibility and reproducibility. The UI is great for quick experiments and prototyping.
:::

---

Once the datasets are created, you're ready to start the fine-tuning job.
