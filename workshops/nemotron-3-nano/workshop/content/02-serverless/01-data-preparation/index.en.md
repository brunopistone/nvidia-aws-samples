---
title: "Data preparation"
weight: 1
---

## What you will learn

In this lesson you will turn annotated non-disclosure agreements into supervised examples, inspect the relationship between a verdict and its evidence, validate the training and evaluation schemas, and register three datasets for the managed workflow. The unit of supervision is one complete contract, not one isolated clause.

::alert[Open [01-serverless-workshop/1-prepare-data.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/1-prepare-data.ipynb). Run the cells from that directory after completing [Studio setup](/01-prerequisites/3-sagemaker/). The Python excerpts below use the notebook's variables and helpers.]{type="info"}

## The business task: a verdict is not enough

A contract-review assistant applies the same 17-point checklist to every NDA. For each hypothesis it must decide whether the agreement supports it, conflicts with it, or does not address it. A human reviewer must also be able to inspect the supporting clauses. A fluent explanation without traceable evidence does not meet that requirement, and valid JSON alone does not establish that a legal interpretation is correct.

[ContractNLI](https://stanfordnlp.github.io/contract-nli/) supplies the documents and expert annotations under CC-BY-4.0. This workshop retains its document-level splits: 423 training contracts, 61 development contracts, and 123 test contracts. The development split becomes `val` in the workshop. With 17 checklist items per contract, the training set represents 7,191 annotated decisions, while the test set represents 2,091 decisions. Those are multiple decisions from shared documents, not independent documents.

| Verdict         | Interpretation                                | Evidence expected by the task |
| --------------- | --------------------------------------------- | ----------------------------- |
| `Entailment`    | The contract states or implies the hypothesis | Supporting span identifiers   |
| `Contradiction` | The contract conflicts with the hypothesis    | Conflicting span identifiers  |
| `NotMentioned`  | The contract does not address the hypothesis  | An empty list                 |

An exception can change a verdict. For example, a blanket prohibition on disclosure and a prohibition that explicitly permits disclosure to employees do not support the same checklist answers. Do not treat absence of an exact phrase as proof that a hypothesis is unmentioned.

## 1. Load and inspect the raw documents

Run the dependency cell separately as a Jupyter magic, then run the notebook's session setup. Confirm the printed execution role, S3 bucket, and Region before any upload.

```ipython
%pip install -r requirements.txt
```

```python
import contractnli as C

C.ensure_dataset("./data")
train_docs, labels = C.load("train")
dev_docs, _ = C.load("dev")
test_docs, _ = C.load("test")

print(f"train {len(train_docs)} contracts | dev {len(dev_docs)} | test {len(test_docs)}")
print(f"checklist items: {len(labels)}")
```

The helper downloads the archive to `./data/contract-nli`. `C.load` reads `documents` and `labels` from the selected split. If you configure `CONTRACTNLI_DIR`, ensure that it points to the directory the loader should read; changing a download target alone does not change that environment-based loader path.

| Raw field or helper                        | Actual content                                 | Transformation                       |
| ------------------------------------------ | ---------------------------------------------- | ------------------------------------ |
| `doc["text"]`                              | Complete contract text                         | Kept as the source for clauses       |
| `doc["spans"]`                             | Character-offset pairs `(start, end)`          | Sliced and numbered by `C.doc_spans` |
| `doc["annotation_sets"][0]["annotations"]` | Hypothesis keys mapped to `choice` and `spans` | Returned by `C.gold_for`             |
| `labels[key]`                              | `hypothesis` and `short_description`           | Rendered into the fixed checklist    |

`C.doc_spans` collapses whitespace within a span but retains the original enumeration index. If a span is empty it is omitted from the rendered text without renumbering the other spans. Citations therefore refer to dataset identifiers, not a newly generated contiguous list of clauses.

```python
doc_example = train_docs[0]
for number, text in C.doc_spans(doc_example)[:3]:
    print(number, text)

first_key = list(labels)[0]
print(labels[first_key])
print(C.gold_for(doc_example)[first_key])
```

Expect numbered text and an annotation shaped like `{"choice": "NotMentioned", "spans": []}`, with the actual verdict and spans determined by the selected document. Do not replace the dataset's checklist with `nda-1` through `nda-17`: its identifiers are not contiguous and extend to `nda-20`.

### Read one annotation before building hundreds

The notebook selects the second-shortest test NDA for inspection. In its Navidec example, span `[3]` prohibits disclosure to any person or company without prior written permission. The expert annotations mark employee sharing (`nda-5`) and third-party sharing (`nda-7`) as `Contradiction`, both citing `[3]`. The broader scope of confidential material makes `nda-2` a `Contradiction` citing `[3, 4]`; the use restriction in `[4]` supports `nda-4` as `Entailment`.

This excerpt is from that annotated target, not a generated model response and not a complete training record:

```json
{
  "nda-2": { "label": "Contradiction", "evidence": [3, 4] },
  "nda-7": { "label": "Contradiction", "evidence": [3] },
  "nda-5": { "label": "Contradiction", "evidence": [3] },
  "nda-4": { "label": "Entailment", "evidence": [4] }
}
```

One clause can justify several decisions. Keep the entire document in the prompt so the model can consider interactions and exceptions across clauses. Keep the provided document-level split as well: training on one clause from an NDA and testing on another clause from the same NDA would weaken the held-out comparison.

## 2. Build the prompt and target

The serverless track uses `C.build_prompt(doc, labels)`. It renders the instruction, numbered contract, checklist, required JSON shape, and the final `/no_think` string. The string requests a direct answer; it is not a JSON validator, a token limit, or a universal guarantee that reasoning text will never appear. Later, inspect the actual generated response and evaluation prompt.

```python
import json

label_keys = list(labels.keys())

def gold_json(doc):
    g = C.gold_for(doc)
    return json.dumps({
        k: {"label": g[k]["choice"], "evidence": list(g[k]["spans"])}
        for k in label_keys if k in g
    })

prompt = C.build_prompt(doc_example, labels)
target = gold_json(doc_example)
record = {"prompt": prompt, "completion": target}
print(prompt)
print(json.dumps(json.loads(target), indent=2))
```

`gold_json` renames `choice` to `label` and `spans` to `evidence`; it does not ask another model to create supervision. The completion is a **string containing JSON**, not a nested object in the outer training record. Serializing the outer record escapes the quotes in that string. This double layer is intentional: parse the JSONL line first, then parse its completion to inspect the verdict object.

The helper also exposes `build_system`, `build_user`, and `build_messages`. Those construct a different representation: the checklist is in the system turn before the user contract. Sharing a module does not make these two representations byte-identical. Changing `C.INSTRUCTION` in one kernel also does not persist a change to another notebook or update the separate `SYSTEM` template. Treat prompt changes as versioned changes across all actual callers.

## 3. Create all three record sets

Train and validation contain string-valued `prompt` and `completion`. The managed custom evaluator consumes the test set in `gen_qa` form, with `query` and `response`. Here `query` is the same constructed prompt under a different field name, and `response` is the reference answer. There is no separate `system` column in these test records.

```python
def make_records(docs):
    return [{"prompt": C.build_prompt(d, labels), "completion": gold_json(d)}
            for d in docs]

def make_test_records(docs):
    return [{"query": C.build_prompt(d, labels), "response": gold_json(d)}
            for d in docs]

records = {
    "train": make_records(train_docs),
    "val": make_records(dev_docs),
    "test": make_test_records(test_docs),
}

for split, docs in (("train", train_docs), ("val", dev_docs), ("test", test_docs)):
    prompt_field = "query" if split == "test" else "prompt"
    target_field = "response" if split == "test" else "completion"
    for rec, doc in zip(records[split], docs):
        assert set(rec) == {prompt_field, target_field}
        assert rec[prompt_field] == C.build_prompt(doc, labels)
        assert json.loads(rec[target_field])
```

The notebook checks field names, prompt equality, and non-empty parseable targets. Add the following local validation checkpoint when inspecting your records. It strengthens those checks without changing the data or calling an AWS API.

```python
for split, docs in (("train", train_docs), ("val", dev_docs), ("test", test_docs)):
    target_field = "response" if split == "test" else "completion"
    assert len(records[split]) == len(docs)
    for rec, doc in zip(records[split], docs):
        answer = json.loads(rec[target_field])
        assert set(answer) == set(labels)
        available_spans = {i for i, _ in C.doc_spans(doc)}
        for item in answer.values():
            assert item["label"] in C.LABELS
            assert isinstance(item["evidence"], list)
            assert all(type(i) is int for i in item["evidence"])
            assert set(item["evidence"]) <= available_spans
            if item["label"] == "NotMentioned":
                assert item["evidence"] == []
```

### Sequence length is a separate validation

A contract can fit as text and still exceed a training recipe's token limit once the answer and formatting are included. Character count divided by four is only a rough estimate. The current serverless notebooks contain conflicting prose about length caps and hard-coded retained-record counts; they do not measure the current 30B-A3B recipe's effective training set. Inspect the resolved trainer hyperparameters in the next lesson and the actual preprocessing logs. Do not report the notebook's fixed retained-count arithmetic as a measurement, and do not assume a recipe drops records when it actually truncates them, or vice versa.

## 4. Write, upload, and register

Run the notebook's staging cell to write one outer JSON object per line at `./sft_data/<split>/dataset.jsonl`. It recreates `./sft_data`, so keep personal files out of that staging directory. Raw ContractNLI data lives separately under `./data`.

The upload portion uses the session's optional default prefix and the configured `DATASET_PREFIX`:

```python
from config import DATASET_PREFIX

input_path = (f"{default_prefix}/datasets/{DATASET_PREFIX}" if default_prefix
              else f"datasets/{DATASET_PREFIX}")
s3_paths = {}
for name in records:
    key = f"{input_path}/{name}/dataset.jsonl"
    s3_client.upload_file(str(local / name / "dataset.jsonl"), bucket_name, key)
    s3_paths[name] = f"s3://{bucket_name}/{key}"
```

`local` is the notebook's `pathlib.Path("./sft_data")`. The resulting URI shape is `s3://<bucket>/<optional-prefix>/datasets/contractnli-nda-review/<split>/dataset.jsonl`. Bracketed placeholders describe a path shape, not literal S3 key characters.

```python
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique

def register(name, source, technique=None):
    kwargs = dict(name=name, source=source, wait=True)
    if technique is not None:
        kwargs["customization_technique"] = technique
    return DataSet.create(**kwargs)

training_dataset = register(f"{DATASET_PREFIX}-train", s3_paths["train"],
                            CustomizationTechnique.SFT)
val_dataset = register(f"{DATASET_PREFIX}-val", s3_paths["val"],
                       CustomizationTechnique.SFT)
test_dataset = register(f"{DATASET_PREFIX}-test", s3_paths["test"])
```

Registration gives later notebooks named dataset objects rather than requiring the data-preparation kernel to remain alive. Wait for imports to become available, and inspect them in Studio under **Assets**, **Datasets**. A successful import is not a substitute for the per-record checks above.

::alert[Both tracks write train and validation to these same default keys, but Track 2 writes conversational arrays. Separate notebook directories do not isolate S3. Use one track per setup or separate the bucket/prefix and resource configuration consistently before uploading. Registry version metadata alone does not preserve old bytes when the notebook overwrites a fixed object key.]{type="warning"}

## Checkpoint and troubleshooting

| Check or symptom                                 | What to inspect before continuing                                                                                |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| All three imports available                      | Save the dataset names, versions, and exact S3 URIs used                                                         |
| Missing dataset files                            | Confirm working directory and `CONTRACTNLI_DIR`; inspect the download error rather than bypassing TLS validation |
| Import failure or wrong schema                   | Parse the first and subsequent JSONL records; verify string fields, not Track 2 message arrays                   |
| Unexpected checklist keys                        | Use `labels.keys()` from the dataset, not a numeric range                                                        |
| S3 access denied                                 | Check object write permissions and bucket/prefix selection; a registered dataset does not grant S3 access        |
| Repeated preparation changes an older experiment | Fixed S3 keys were reused; preserve immutable input objects for reproducible runs                                |

You are ready for [Fine-tuning](/02-serverless/02-fine-tuning/) when the three named datasets are available, every target passes validation, and you can explain where a selected verdict's citation comes from.
