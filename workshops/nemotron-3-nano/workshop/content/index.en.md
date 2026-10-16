---
title: "Fine-tune NVIDIA Nemotron 3 models on Amazon SageMaker AI"
weight: 0
---

## Build a contract-review assistant that cites its evidence

This workshop teaches supervised fine-tuning of NVIDIA Nemotron 3 models on Amazon SageMaker AI. The primary journey uses **Nemotron 3 Nano 4B with SageMaker Training jobs**. An **optional Serverless model customization** extension uses Nemotron 3 Nano 30B-A3B and a managed recipe. Both use ContractNLI: a model reads a non-disclosure agreement, applies a fixed checklist of 17 legal hypotheses, and returns a verdict plus numbered evidence spans for every item.

Start with [SageMaker Training jobs: 4B](/03-training-jobs/) after prerequisites. It is a standalone journey: prepare data, train, deploy, evaluate, and clean up. You do not need to run the serverless lessons first. The [optional Serverless model customization](/02-serverless/) pages remain available for learners interested in managed recipes, registry lineage, and managed evaluation. The source directories retain their historical numeric prefixes; those numbers are not the recommended learning order.

The goal is not simply fluent text or valid JSON. A reviewer needs a correct decision and an inspectable citation. A model that claims a protection exists when the agreement is silent can be more harmful than one that openly returns an incomplete answer. This educational workflow requires human legal review and is not a substitute for legal advice.

## Learning objectives

By the end of the primary journey, you should be able to:

- Transform document-level annotations into conversational supervised records and preserve held-out documents.
- Explain supervised fine-tuning, Low-Rank Adaptation, precision, adapter targets, token budgets, and completion-only loss.
- Submit and monitor a SageMaker Training job running TRL, and inspect its metrics in the configured MLflow App.
- Trace training inputs and configuration to the merged S3 artifact and its serving resources.
- Verify deployment identity, requests, output schema, and endpoint progress using Studio and CloudWatch.
- Interpret verdict accuracy, citation F1, schema compliance, and judge coverage without treating a smoke test as a benchmark.
- Separate full-response latency from streaming time to first generated content and report measurement conditions.
- Identify and clean up independently billed training, evaluation, hosting, Studio, and retained storage resources.

## Understand the output contract

Each checklist key maps to a verdict and an evidence list:

| Label           | Meaning                                    | Evidence                            |
| --------------- | ------------------------------------------ | ----------------------------------- |
| `Entailment`    | The contract supports the hypothesis       | Supporting clause/span identifiers  |
| `Contradiction` | The contract conflicts with the hypothesis | Conflicting clause/span identifiers |
| `NotMentioned`  | The contract does not address it           | Empty list                          |

### Read the actual Navidec example

The following clauses and hypotheses come from the worked example in [1-prepare-data.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/1-prepare-data.ipynb). This is an excerpt of an annotated ContractNLI document, not invented contract text and not a generated model result. Other spans and checklist items are omitted here; the full record includes them.

```text
[0] NAVIDEC, INCORPORATED
[1] TRADE SECRET/NON-DISCLOSURE AGREEMENT
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

nda-2: Confidential Information shall only include technical information.
nda-7: Receiving Party may share some Confidential Information with some
       third-parties (including consultants, agents and professional advisors).
nda-5: Receiving Party may share some Confidential Information with some of
       Receiving Party's employees.
```

The corresponding expected JSON excerpt is:

```json
{
  "nda-2": { "label": "Contradiction", "evidence": [3, 4] },
  "nda-7": { "label": "Contradiction", "evidence": [3] },
  "nda-5": { "label": "Contradiction", "evidence": [3] }
}
```

Span `[3]` prohibits disclosure without prior written permission. The annotation marks employee and third-party sharing as contradicted, citing that clause. Spans `[3, 4]` describe material beyond technical information, supporting the annotated contradiction for `nda-2`. One clause can justify several decisions, and exceptions elsewhere in a contract can change the interpretation. Inspect the dataset judgment rather than treating it as universal legal advice.

The full target contains all 17 original keys, which are not consecutively numbered. ContractNLI's document-level split is 423 training, 61 development, and 123 test contracts. Keeping entire documents in one split prevents training on one part of an NDA and testing on another. The test set's 2,091 decisions share 123 documents, so they are not 2,091 independent contracts.

## What supervised fine-tuning changes

![Supervised fine-tuning: an instruction and contract condition a model, gold answer tokens supply the loss, and only LoRA adapters are updated before export and held-out evaluation.](/static/images/general/sft-contractnli.svg)

Supervised fine-tuning (SFT) trains on demonstrations of the desired response. Each example pairs an input with an expert completion. During training, the model predicts successive completion tokens conditioned on the prompt and earlier gold completion tokens. The loss rewards probability assigned to those target tokens. At inference, no gold answer is supplied: the model conditions on its own generated prefix. That difference is why low training loss alone does not establish reliable generation.

In this recipe the prompt contains the standing instruction/checklist and the contract, while the completion contains the verdict-and-evidence JSON. Completion-only loss masks prompt and padding positions rather than teaching the model to reproduce the contract. Template boundaries, truncation, EOS, and tokenization affect which tokens actually receive supervision. LoRA freezes base weights and learns low-rank updates to selected modules. It reduces trainable parameters and optimizer state, not the requirement to hold base weights and activations. BF16 LoRA is not four-bit QLoRA.

Fine-tuning is suitable to test when repeated examples should change task behavior or output consistency. It does not replace retrieval for changing facts, licensing review, or human judgment. Establish a baseline, iterate with validation data, and evaluate on held-out documents before asserting improvement. The primary lessons explain the exact record structure, adapter targets, and measurements rather than assuming SFT will outperform prompting.

## Primary journey and optional extension

| Dimension          | [SageMaker Training jobs: 4B](/03-training-jobs/)                                              | [Optional: Serverless model customization](/02-serverless/) |
| ------------------ | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Model              | Nemotron 3 Nano 4B BF16                                                                        | Nemotron 3 Nano 30B-A3B BF16                                |
| Model identifier   | Hugging Face ID from `config.py`                                                               | JumpStart ID from `config.py`                               |
| Training interface | `ModelTrainer` running TRL `SFTTrainer` and PEFT                                               | SageMaker `SFTTrainer`, managed LoRA recipe                 |
| Input shape        | Conversational `prompt`/`completion` with template options                                     | String `prompt`/`completion`; test `query`/`response`       |
| Input handoff      | S3 `train`, `val`, and `config` channels                                                       | AI Registry datasets                                        |
| Training compute | One single-GPU instance, one epoch; five ordered preferences with `ml.g5.2xlarge` first | Selected by the managed recipe |
| Training output    | Merged checkpoint/tokenizer in S3 `model.tar.gz`                                               | Model Package with merged checkpoint prefix                 |
| Hosting | Direct production variant, one instance across four prioritized pools with `ml.g5.xlarge` first | Four-GPU inference component on one `ml.g5.12xlarge` |
| Quality evaluation | Endpoint predictions with local scoring; optional Bedrock judge; shipped references by default | Managed custom scorer and LLM judge                         |

The primary journey exposes training internals, inline YAML, dependencies, model export, and direct serving. Warm-pool retention currently defaults to `KEEP_ALIVE_SECONDS = 0`. The optional extension explores managed recipe configuration and registry-based handoffs. These are alternatives, not consecutive required labs or a controlled comparison of model sizes. They differ in model, prompt representation, recipe, and evaluation implementation. Active-parameter counts do not establish task quality or the complete storage requirement for weights.

::alert[**Use one workflow per setup.** Both configurations default to `datasets/contractnli-nda-review` in the session bucket, including the same train and validation object keys. Their schemas differ. Running one workflow's upload cells over the other can invalidate registered data without changing its name. Separate notebook folders do not isolate S3; isolate bucket/prefix and resource naming consistently before attempting both.]{type="warning"}

## How to use the workshop

The notebooks and helpers live in [aws-samples/generative-ai-on-amazon-sagemaker](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/tree/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai). Open them in a SageMaker AI Studio JupyterLab space and execute the primary journey from `02-smtj-workshop/`. The pages explain transformations, annotated code, checks, and failure modes; the linked notebooks remain the execution source. Keep each evaluation notebook beside its own `evaluation_utils.py`: the two track-specific helpers have different APIs. The evaluation lessons explain which calls remain in the notebooks and which supporting implementations live in these modules.

| Stage                                                      | What to retain before proceeding                                                                                           |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [Prerequisites](/01-prerequisites/)                        | Account, Region, role, MLflow reuse/create permissions (configured ARN recommended if precreated), and quota/access checks |
| [Data preparation](/03-training-jobs/01-data-preparation/) | Schema, split counts, input locations, and prompt/template configuration                                                   |
| [Fine-tuning](/03-training-jobs/02-fine-tuning/)           | Job identity, resolved configuration, MLflow run, and artifact identity                                                    |
| [Deployment](/03-training-jobs/03-deployment/)             | Model, EndpointConfig, Endpoint, image, and artifact URI                                                                   |
| [Evaluation](/03-training-jobs/04-evaluation/)             | Prediction/reference pairs, scorer definition, measurement metadata, and coverage                                          |
| [Clean Up](/04-cleanup/)                                   | Confirmation that unwanted compute and resources are no longer active                                                      |

Execute `1-prepare-data.ipynb`, `2-fine-tune-llm.ipynb`, `3-deployment.ipynb`, then `4-evaluation.ipynb`. Evaluation switches default to shipped references, so analysis without live evaluation does not measure your deployed model. Keep hosting running for fresh prediction collection, not merely for reading references or judging saved answers. Hosting cleanup is at the end of notebook 4 and runs independently of the evaluation switches; do not use Run All without reviewing that cell. Some source Markdown describes historical settings or results; use executable configuration and distinguish reference results from your measurements.

## Cost, security, and evidence

Training, evaluation inference, Bedrock judging, endpoint hosting, Studio compute, MLflow, warm pools, and storage are separate cost categories. Serverless **training** does not make the real-time **endpoint** serverless. Stopping a kernel does not stop a remote job, and a local timeout does not delete an endpoint. No fixed runtime, cost reduction, or quality improvement is promised. Use [SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/) with your Region and pricing date rather than a generic cost estimate. The primary journey's [Instance Preferences](/03-training-jobs/02-fine-tuning/#instance-preferences) and [Instance Pools](/03-training-jobs/03-deployment/#instance-pools) select capacity alternatives, not guaranteed cheapest hardware. Price the actual selected training type and allocated hosting pool types and counts, not the first candidates alone.

The [Bedrock Custom Model Import supported-architecture list](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html#model-customization-import-model-architecture), checked on 2026-10-14, does not list NemotronH. This workshop therefore does not offer Custom Model Import as a deployment path for these artifacts. Native model availability in Bedrock is a separate question from importing customized weights. Optional Bedrock judging consumes saved responses, not an imported Nemotron checkpoint.

Use approved data and a dedicated environment. Review ContractNLI and model licenses; never upload confidential contracts or credentials to a temporary event account. Apply IAM permissions to actual resources and distinguish role trust from permission to invoke a service. Preserve data/configuration/artifact lineage so results can be audited rather than inferred from a familiar resource name.

:button[Start prerequisites]{href="/01-prerequisites/"}
