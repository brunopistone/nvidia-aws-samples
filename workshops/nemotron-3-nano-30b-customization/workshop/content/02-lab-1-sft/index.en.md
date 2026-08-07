---
title: "Lab 1️⃣: Supervised Fine-Tuning (SFT)"
weight: 3
---

In this lab, you will learn how to fine-tune **Qwen3 4B** using Serverless Supervised Fine-Tuning on Amazon SageMaker AI.

SFT is the most straightforward customization technique — you provide prompt-response pairs and the model learns to produce the desired outputs. It's ideal for teaching a model new tasks like classification, extraction, or structured reasoning.

:image[Supervised Fine-Tuning concept]{src="/static/images/lab-1-sft/sft_concept.jpeg" height=384}

## What you will learn

1. **Prepare your dataset** - Format data for SFT and create SageMaker AI Datasets
2. **Run a serverless fine-tuning job** - Use the SFTTrainer API with LoRA
3. **Evaluate your model** - Register a deterministic custom scorer and run the managed evaluation pipeline
4. **Deploy your model** - Create a SageMaker real-time endpoint

## Model and Dataset

| Component      | Details                                                                                                                      |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Base Model** | Qwen3 4B (`huggingface-reasoning-qwen3-4b`), set in `config.py`                                                              |
| **Dataset**    | [ContractNLI](https://stanfordnlp.github.io/contract-nli/) (Koreeda & Manning, Findings of EMNLP 2021), CC-BY-4.0            |
| **Technique**  | Supervised Fine-Tuning with LoRA                                                                                             |
| **Use Case**   | NDA checklist review with evidence citation                                                                                   |

## The task in one sentence

Given a non-disclosure agreement, answer a **fixed 17-point legal checklist** about it, and for each answer **cite the clause numbers** that justify it.

Three possible verdicts per item:

| Verdict           | Meaning                                                    |
| ----------------- | ---------------------------------------------------------- |
| `Entailment`      | the contract states or implies the item is true            |
| `Contradiction`   | the contract says something that conflicts with it         |
| `NotMentioned`    | the contract does not address the subject at all           |

### The input

One request is **a single string**: the instruction, the contract as numbered spans, then the checklist, then the required output shape. This is the real template, with the contract truncated and the checklist showing 3 of its 17 items:

```
You are a contract review assistant. You review a non-disclosure agreement (NDA)
against a fixed checklist of 17 legal hypotheses.

For EACH hypothesis, decide:
- "Entailment": the contract states or implies the hypothesis is true.
- "Contradiction": the contract states something that conflicts with the hypothesis.
- "NotMentioned": the contract does not address it.

Also cite the span numbers that justify the decision (the exact spans a lawyer
would point to). Cite spans only for Entailment or Contradiction; use an empty
list for NotMentioned. Read exceptions and carve-outs carefully: a clause with an
exception may contradict a hypothesis stated absolutely.

CONTRACT (numbered spans):
[0] NAVIDEC, INCORPORATED
[1] TRADE SECRET/NON-DISCLOSURE AGREEMENT
[2] In consideration of the mutual promises made herein, as well as the agreement
    between Navidec, Incorporated and _______ , the parties hereby agree as follows:
[3] _______ , agrees that, in consideration for being shown or told about certain
    trade secrets or property belonging to Navidec, Incorporated, _______ , shall
    not disclose or cause to be disclosed, disseminated or distributed any
    information concerning said trade secret or property to any person, entity,
    business or other individual or company without the prior written permission
    of Navidec, Incorporated.
[4] Further, _______ , agrees not to use, either directly or indirectly any of the
    material, ideas, objects or portions thereof of said trade secret or property
    disclosed by Navidec, Incorporated in any manner whatsoever without the prior
    written consent of Navidec, Incorporated.
    ... (remaining spans)

CHECKLIST:
nda-2: Confidential Information shall only include technical information.
       (None-inclusion of non-technical information)
nda-7: Receiving Party may share some Confidential Information with some
       third-parties (including consultants, agents and professional advisors).
       (Sharing with third-parties)
nda-5: Receiving Party may share some Confidential Information with some of
       Receiving Party's employees. (Sharing with employees)
    ... (17 items total, always in the same order)

Respond with JSON only, no other text:
{"nda-1": {"label": "Entailment|Contradiction|NotMentioned", "evidence": [span numbers]}, ...}
Include an entry for every hypothesis key listed above.

/no_think
```

Note the carve-out sentence in the instruction: **carve-outs matter**. That single sentence is what the example below turns on. The checklist sits *after* the contract so the last thing the model reads before answering is what it is being asked.

:::alert{header="That last line is not decoration" type="warning"}
The base model is a reasoning model. Asked to deliberate over 17 hypotheses it produces thousands of tokens inside `<think>` and can exhaust its generation budget before emitting any JSON. `/no_think` is the documented soft switch that turns thinking off, and it goes at the end of the prompt.

It is part of every split's prompt and of the deployed request, so training, evaluation and serving all send the same string. It is also the difference between a base model that looks broken and one that scores a respectable 64.8 — see [Evaluation](03-evaluation).
:::

Prompt length, counted with the Qwen3 4B tokeniser over the training split: median **2,870 tokens**, mean **3,116**, longest **14,016**. Everything except the contract — instruction, checklist, required output shape — is a fixed 3,234 characters, identical in every record. The whole contract goes in every time, because any of the 17 items could be decided by any clause.

### The expected output

Strict JSON. One key per checklist item, in the checklist's order, each with a verdict and a list of span numbers. This is the complete, exact answer for the contract above — all 17 items, nothing omitted:

```json
{"nda-11": {"label": "NotMentioned",  "evidence": []},
 "nda-16": {"label": "NotMentioned",  "evidence": []},
 "nda-15": {"label": "NotMentioned",  "evidence": []},
 "nda-10": {"label": "NotMentioned",  "evidence": []},
 "nda-2":  {"label": "Contradiction", "evidence": [3, 4]},
 "nda-1":  {"label": "NotMentioned",  "evidence": []},
 "nda-19": {"label": "NotMentioned",  "evidence": []},
 "nda-12": {"label": "NotMentioned",  "evidence": []},
 "nda-20": {"label": "NotMentioned",  "evidence": []},
 "nda-3":  {"label": "NotMentioned",  "evidence": []},
 "nda-18": {"label": "NotMentioned",  "evidence": []},
 "nda-7":  {"label": "Contradiction", "evidence": [3]},
 "nda-17": {"label": "NotMentioned",  "evidence": []},
 "nda-8":  {"label": "NotMentioned",  "evidence": []},
 "nda-13": {"label": "NotMentioned",  "evidence": []},
 "nda-5":  {"label": "Contradiction", "evidence": [3]},
 "nda-4":  {"label": "Entailment",    "evidence": [4]}}
```

### Why those are the right answers

Take `nda-5`: *"Receiving Party may share some Confidential Information with some of Receiving Party's employees."*

Span [3] forbids disclosure **"to any person, entity, business or other individual or company"** with no exception for employees. Most NDAs carve one out; this one does not. So the checklist statement is not merely unaddressed — it is **contradicted**, and span [3] is the proof.

`nda-7` (sharing with third parties) is contradicted by the same span. `nda-2` (*"Confidential Information shall only include technical information"*) is contradicted by [3] and [4] together, which cover "material, ideas, objects" and "trade secrets or property" far beyond technical information. And `nda-4` (*"shall not use Confidential Information for any purpose other than the purposes stated in the Agreement"*) is **entailed** by [4].

Two things to notice, because they define the difficulty:

- **One clause drives several answers.** Span [3] decides three items. This is not retrieval where each question has its own passage.
- **13 of 17 items are `NotMentioned` in this contract.** Short NDAs are silent on most of the checklist. Across the whole test split the skew is milder — 43% `NotMentioned`, 46% `Entailment`, 10% `Contradiction` — but it still means a model that always answers `NotMentioned` scores 43% accuracy without reading anything, while scoring exactly zero on evidence. That is why the evaluation section leads with evidence-F1 instead.

## Why fine-tuning for contract review?

This task shape — a fixed checklist, applied repeatedly to one document type, where the output is a verdict plus a pointer to the text that justifies it — recurs well beyond legal work: vendor security questionnaires, insurance claims adjudication, regulatory filing checks, clinical trial screening, RFP compliance.

Fine-tuning helps because the two failure modes that matter are hard to prompt away:

- Claiming a requirement is satisfied when the document is actually **silent** on it
- Citing the **wrong clause**, so a reviewer following the citation finds nothing

Measured on the full 123-contract held-out set (2,091 checklist decisions):

| Model                            | Accuracy | Evidence-F1 |
| -------------------------------- | -------- | ----------- |
| Base Qwen3 4B                    | 64.8     | 48.9        |
| Claude Sonnet 5 (zero-shot)      | 83.6     | 67.1        |
| **Fine-tuned Qwen3 4B (LoRA)**   | **83.8** | **68.9**    |

![Statistical scoring, 123 held-out contracts](/static/images/lab-1-sft/statistical-scoring-chart.png)

The headline is that a **4B model you fine-tuned yourself reaches Claude Sonnet 5's level** on this task, at roughly 8x lower cost per contract. It finishes marginally ahead on both metrics — call that parity, not a win, for two reasons: the margin is 0.2 points on accuracy, and the two rows are not computed by the same averaging (see [Evaluation](03-evaluation)).

:::alert{header="The base model is not a straw man here" type="info"}
Base Qwen3 4B scores a genuine 64.8 / 48.9 — it returns valid JSON on **99.2%** of contracts, so these are real measurements of what the model already knows rather than a formatting failure. Getting there takes one line in the prompt: `/no_think`, which turns off the reasoning trace that would otherwise consume the generation budget before the JSON appears.

That makes the gain the honest one to quote — **+19 accuracy and +20 evidence-F1 over the same base weights**, from a working starting point rather than from zero.
:::

:::alert{header="Where these numbers come from" type="info"}
This table and the cost figures below are measured on one run using the same settings the lab configures — `max_epochs = 10`, LoRA rank 32, on the 123-contract held-out split. A fine-tune is not bit-reproducible and this training set is small, so expect a few points of movement in either direction; the parity with Sonnet may land as a narrow win or a narrow loss on your run.
:::

Adding examples to the prompt makes the small model **worse** on this task, not better — few-shot demonstrations degrade its contradiction detection, because they teach output shape and shape was never the bottleneck. That is the clearest signal that this is a fine-tuning problem rather than a prompt-engineering one.

### And it is cheaper to run

Quality is only half the argument. Measured on the same reference run, at standard Sonnet pricing ($3 / $15 per 1M tokens):

| | Per contract | 2,000 contracts/month | 20,000 contracts/month |
| --------------------------------------------------- | ---------- | ----- | ----- |
| Claude Sonnet 5                                     | $0.02854   | $57   | $571  |
| **Fine-tuned Qwen3 4B** (Bedrock Custom Model Import) | **$0.00353** | **$9** | **$73** |

Roughly **8x cheaper per contract**, from two sources. Part is simply that a 4B model is small. The rest is output length: Sonnet writes 1,023 output tokens per contract because it explains its reasoning, while the fine-tuned model writes 389 — it was trained to emit only the JSON, and output tokens are the expensive ones.

Against that, the cost of *creating* the model: **under $2 and 20-25 minutes** for the LoRA job in the Fine-Tuning section. At these rates that is repaid after roughly **80 contracts**.

Two qualifications worth stating before you quote any of this to a customer:

- **Below roughly 78 contracts a month, Sonnet is cheaper.** Owning a model means paying for capacity rather than per token, so at trivial volume the fixed costs dominate. Fine-tuning is not automatically the cheap option.
- **The dominant cost is annotation, not compute.** 423 contracts × 17 verdicts with evidence spans is about 7,200 expert judgements. For a customer with no historical review records to mine, that dwarfs every number in the table.

:::alert{header="Note" type="info"}
Prices were verified for `us-east-1` / `us-west-2` at the time of writing and change over time. Re-check before quoting them, and use the [AWS Pricing Calculator](https://calculator.aws/) with the measured token counts — Sonnet 5 uses 4,399 in / 1,023 out per contract, the fine-tuned model 3,126 / 389.
:::

## Lab Structure

| Section                                 | Description                                                              | Duration   |
| --------------------------------------- | ------------------------------------------------------------------------ | ---------- |
| [Data Preparation](01-data-preparation) | Prepare and upload dataset to S3, create SageMaker Datasets              | ~10 min    |
| [Fine-Tuning](02-fine-tuning)           | Launch serverless SFT job with LoRA                                      | 20-25 min  |
| [Evaluation](03-evaluation)             | Baselines, then base + tuned model via the managed evaluation pipeline    | 60-75 min  |
| [Deployment](04-deployment)             | Deploy to SageMaker real-time endpoint                                    | ~10 min    |

:::alert{header="Important" type="warning"}
This workshop is designed to demonstrate serverless model customization, not to produce a production-grade model. The dataset ships 423 training contracts, which is small — but each carries 17 supervised decisions plus evidence spans, so it amounts to roughly 7,200 labelled judgements. You can adapt this codebase to fine-tune the model using a larger dataset as needed.
:::
