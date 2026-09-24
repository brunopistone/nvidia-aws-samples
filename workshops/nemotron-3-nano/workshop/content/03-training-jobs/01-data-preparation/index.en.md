---
title: "Data preparation"
weight: 1
---

## What you will learn

You will map ContractNLI annotations into conversational prompt/completion records, inspect the model's own chat template, carry the reasoning-control flag into each example, measure token lengths before training, and upload only the splits the Training job should consume. The resulting data is intentionally different from the serverless track's string-valued records.

::alert[Open [02-smtj-workshop/1-prepare-data.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/1-prepare-data.ipynb) from its own directory. Complete [Studio setup](/01-prerequisites/3-sagemaker/) first, and check that `config.py` selects `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`.]{type="info"}

## The task and the split boundary

The business task is repeated NDA review against a fixed checklist of 17 hypotheses. The assistant must return a JSON verdict and evidence span identifiers for each item so a reviewer can inspect the relevant text. This is supervised adaptation to a particular output contract and document interpretation task, not general legal certification.

[ContractNLI](https://stanfordnlp.github.io/contract-nli/) provides 423 training documents, 61 development documents, and 123 held-out test documents under CC-BY-4.0. Preserve those document-level partitions. Splitting individual clauses randomly could expose parts of the same agreement during both training and testing. The development set becomes `val`; test remains reserved for the eventual held-out comparison.

| Source annotation | Target field    | Meaning                                            |
| ----------------- | --------------- | -------------------------------------------------- |
| `choice`          | `label`         | `Entailment`, `Contradiction`, or `NotMentioned`   |
| `spans`           | `evidence`      | Dataset span identifiers that support the decision |
| Hypothesis key    | JSON object key | The original checklist key, not a renumbered index |

`Entailment` means the agreement supports the hypothesis. `Contradiction` means it conflicts with the hypothesis. `NotMentioned` means the agreement does not address it and should have an empty evidence list. A broad prohibition can contradict a permission even without repeating the checklist's wording, and a carve-out can change how an otherwise broad clause applies.

## 1. Load documents and inspect evidence

Install notebook-side requirements in a separate magic cell. These dependencies support the client and tokenizer inspection; the training container installs its own `scripts/requirements.txt` later.

```ipython
%pip install -r requirements.txt
```

```python
import contractnli as C

C.ensure_dataset("./data")
train_docs, labels = C.load("train")
dev_docs, _ = C.load("dev")
test_docs, _ = C.load("test")
print(len(train_docs), len(dev_docs), len(test_docs), len(labels))
```

Run the session setup and verify role, bucket, Region, and optional prefix. `C.ensure_dataset` downloads the archive under `./data/contract-nli`; `C.load` reads its document and checklist collections. `CONTRACTNLI_DIR`, if set, controls the loader's directory. A network download error is different from a local path mismatch.

The document's `spans` are character offsets into `text`. `C.doc_spans` slices each span, collapses whitespace, and retains its original numeric identifier. `C.gold_for` unwraps `annotation_sets[0]["annotations"]`. Read one raw item and the text it cites before building the dataset:

```python
doc_example = train_docs[0]
gold_example = C.gold_for(doc_example)
first_key = list(labels)[0]
print(labels[first_key])
print(gold_example[first_key])
for number, text in C.doc_spans(doc_example)[:3]:
    print(f"[{number}] {text}")
```

The notebook's short Navidec example illustrates why citations matter. Its expert annotation marks `nda-5` and `nda-7`, permissions to share with employees and third parties, as contradictions citing span `[3]`. The same clause can decide multiple checklist items. Another verdict, `nda-2`, cites both `[3, 4]`, while `nda-4` is entailed by `[4]`. This is a worked annotation, not a generated result or evidence of model accuracy.

The full checklist has 17 keys but is not numbered consecutively from 1 to 17. Use `labels.keys()` rather than constructing identifiers yourself. One record contains all 17 decisions, so 423 records represent 7,191 labeled decisions but only 423 independently split documents.

## 2. Build conversational supervision

This track trains with **lists of messages** in `prompt` and `completion`. `C.build_messages` produces a system instruction/checklist, a user contract, and an optional assistant target. The notebook puts the first two turns in `prompt` and the last turn in `completion`.

```python
import json

label_keys = list(labels.keys())

def gold_json(doc):
    g = C.gold_for(doc)
    return json.dumps({
        k: {"label": g[k]["choice"], "evidence": list(g[k]["spans"])}
        for k in label_keys if k in g
    })

target = gold_json(doc_example)
turns = C.build_messages(doc_example, labels, completion=target)
record = {
    "prompt": turns[:-1],
    "completion": turns[-1:],
    "chat_template_kwargs": C.CHAT_TEMPLATE_KWARGS,
}
```

| Record field           | Python/JSON type                     | Role in training                                  |
| ---------------------- | ------------------------------------ | ------------------------------------------------- |
| `prompt`               | List/array of two message objects    | Context: system checklist plus user contract      |
| `completion`           | List/array of one assistant message  | Supervised answer containing the gold JSON string |
| `chat_template_kwargs` | Object with `enable_thinking: false` | Extra options forwarded to the chat template      |

The following is an abbreviated **schema illustration**, not a full training example. The real completion includes every checklist key and the real prompt contains the whole contract and checklist.

```json
{
  "prompt": [
    { "role": "system", "content": "Instruction and complete checklist" },
    {
      "role": "user",
      "content": "CONTRACT (numbered spans):\n[0] Contract text"
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "{\"nda-5\": {\"label\": \"Contradiction\", \"evidence\": [3]}}"
    }
  ],
  "chat_template_kwargs": { "enable_thinking": false }
}
```

There are two serialization layers: a JSONL line is one record, while the assistant's `content` is a string that itself contains JSON. Do not replace `completion` with that decoded object or serialize the entire message list into a single string. TRL needs the message structure to apply the tokenizer's chat template.

### Completion-only loss

The recipe explicitly sets `dataset_format: "prompt_completion"` and `completion_only_loss: true`. The script preserves these columns and passes them to TRL. The intended loss boundary excludes the prompt and supervises the answer, rather than training the model to copy the contract. In conceptual terms, only answer-token positions contribute to the language-model loss:

\[L=-\frac{1}{|T|}\sum*{t\in T}\log p*{\theta}(y*t\mid x,y*{<t})\]

Here \(x\) is the prompt and \(T\) is the set of supervised completion positions. Padding and prompt positions must be masked appropriately. This is why the prompt/completion boundary, special tokens, and truncation behavior are training decisions, not merely file formatting.

## 3. Carry reasoning controls through the template

The local helper defines `C.CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}`. This option is not part of the user text. TRL forwards the per-record dictionary when rendering the template, and the deployment request later passes the same dictionary to vLLM. A plain `/no_think` suffix from the serverless track is not a substitute for this explicit configuration.

With the configured template, the notebook checks for an already-closed `<think></think>` block. That verifies a rendering choice, not a guarantee that every generated answer will be complete or correct. A model can still exceed its output budget, stop early, or produce an invalid verdict object.

`C.build_prompt` in this track is an inspection helper for a flat string; the actual training and serving messages come from `C.build_messages`. The latter uses the `SYSTEM` and `USER` templates, not the inspection-only `INSTRUCTION`. Editing `C.INSTRUCTION` in a kernel therefore does not update the training message wording. Keep actual message construction and tokenizer/template versions aligned when experimenting.

```python
def make_records(docs):
    out = []
    for doc in docs:
        turns = C.build_messages(doc, labels, completion=gold_json(doc))
        out.append({
            "prompt": turns[:-1],
            "completion": turns[-1:],
            "chat_template_kwargs": C.CHAT_TEMPLATE_KWARGS,
        })
    return out

records = {
    "train": make_records(train_docs),
    "val": make_records(dev_docs),
    "test": make_records(test_docs),
}
```

All three splits share the same schema. The following notebook checks validate that the request and control flag survive record construction:

```python
for split, docs in (("train", train_docs), ("val", dev_docs), ("test", test_docs)):
    for rec, doc in zip(records[split], docs):
        assert set(rec) == {"prompt", "completion", "chat_template_kwargs"}
        assert rec["prompt"] == C.build_messages(doc, labels)
        assert rec["completion"][0]["role"] == "assistant"
        assert json.loads(rec["completion"][0]["content"])
        assert rec["chat_template_kwargs"] == {"enable_thinking": False}
```

Also check that every decoded assistant target contains exactly `set(labels)`, uses allowed labels, and cites integer identifiers returned by `C.doc_spans(doc)`. Parseability alone does not check those conditions, and an object with 17 unrelated keys is not a complete checklist answer.

## 4. Measure token lengths with the real tokenizer

The notebook downloads the configured model's tokenizer, not its model weights, and renders complete prompt-plus-completion records. Its `MAX_LENGTH = 8192` should match the **inline `args.yaml` written by notebook 2**. There is no checked-in `scripts/args.yaml` to edit in the current source directory.

```python
from transformers import AutoTokenizer
from config import BASE_MODEL_ID

MAX_LENGTH = 8192
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
lengths = []
for rec in records["train"]:
    tokens = tokenizer.apply_chat_template(
        rec["prompt"] + rec["completion"],
        tokenize=True,
        return_dict=False,
        **rec["chat_template_kwargs"],
    )
    lengths.append(len(tokens))

lengths.sort()
over = sum(n > MAX_LENGTH for n in lengths)
print(f"train records: {len(lengths)}")
print(f"median: {lengths[len(lengths) // 2]}")
print(f"longest: {lengths[-1]}")
print(f"over max_length: {over}")
```

Keep `return_dict=False`, as in the current notebook, so `len(tokens)` counts the returned token IDs rather than the keys of a Transformers `BatchEncoding`. These printed counts are measurements to obtain from your environment, not fixed workshop results. The notebook-side Transformers dependency is a lower bound, whereas training pins a specific version. Record the resolved tokenizer and library version; if rendering differs, repeat the inspection with the training-compatible environment before assuming identical boundaries.

```python
rendered_text = tokenizer.apply_chat_template(
    records["train"][0]["prompt"] + records["train"][0]["completion"],
    tokenize=False,
    **records["train"][0]["chat_template_kwargs"],
)
assert "<think></think>" in rendered_text
```

The configured TRL path truncates over-limit sequences rather than implementing the serverless notebook's described preprocessing filter. A truncated answer can lose some checklist entries; a very long prompt can leave little or no supervised answer within the window. Inspect over-limit examples and the resulting completion boundary before changing `max_length`. Increasing the cap also raises memory requirements. The script does not auto-calculate a length for `prompt_completion`, so the explicit recipe value is important.

## 5. Upload train and validation, retain test locally

Run the staging cell to write JSONL files under `./sft_data`. It removes that staging directory on reruns, so do not store unrelated files there. The raw archive under `./data` remains separate. The upload portion is:

```python
from config import DATA_PREFIX

input_path = f"{default_prefix}/{DATA_PREFIX}" if default_prefix else DATA_PREFIX
train_dataset_s3_path = f"s3://{bucket_name}/{input_path}/train/dataset.jsonl"
val_dataset_s3_path = f"s3://{bucket_name}/{input_path}/val/dataset.jsonl"

for name in ("train", "val"):
    s3_client.upload_file(
        str(local / name / "dataset.jsonl"),
        bucket_name,
        f"{input_path}/{name}/dataset.jsonl",
    )
```

`local` is the notebook's staging `Path`. The same cell copies the held-out records to `./tmp/test.jsonl` and removes staging. No test object is uploaded and no AI Registry dataset is registered.

| Artifact                                                                              | Consumer                                                                             |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `s3://<bucket>/<optional-prefix>/datasets/contractnli-nda-review/train/dataset.jsonl` | Training job `train` channel                                                         |
| Corresponding `val/dataset.jsonl`                                                     | Training job `val` channel                                                           |
| `./tmp/test.jsonl`                                                                    | Preserved conversational test records; notebook 4 reloads raw test documents instead |

::alert[The current `4-evaluation.ipynb` implements endpoint scoring, but does not read `./tmp/test.jsonl`. It calls `C.ensure_dataset("./data")` and `C.load("test")`, then rebuilds prompts and gold references from raw ContractNLI using this track's helpers. Preserve the conversational test file for inspection and keep raw data/helper versions consistent. After training, run `3-deployment.ipynb`, then follow [Evaluation](/03-training-jobs/04-evaluation/) to choose shipped references or fresh predictions.]{type="info"}

The default S3 train/validation keys are shared with Track 1, but their schemas differ. Do not overwrite the other track's registered data. A different notebook directory or a new kernel does not create a new S3 namespace; isolate configuration consistently before running both tracks.

## Checkpoint and troubleshooting

| Symptom                                         | Targeted check                                                                                     |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `config` or helper imports resolve unexpectedly | Check `config.__file__` and `C.__file__`, then restart in the correct track directory              |
| Reasoning flag assertion fails                  | Inspect actual tokenizer/template version and per-record dictionary before training                |
| Lengths exceed the cap                          | Inspect where the answer starts and what truncation removes; do not substitute character estimates |
| Data exists but training sees strings           | Another track may have overwritten the shared S3 keys                                              |
| Test dataset missing from AI Registry           | Expected here; this notebook deliberately creates no registry entries                              |

You are ready for [Fine-tuning](/03-training-jobs/02-fine-tuning/) when the two S3 files have the conversational schema, the local test file is preserved, and your token-length inspection agrees with the inline recipe you will submit.
