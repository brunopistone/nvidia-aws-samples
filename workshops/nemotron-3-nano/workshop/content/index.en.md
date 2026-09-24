---
title: "Fine-tune NVIDIA Nemotron 3 models on Amazon SageMaker AI"
weight: 0
---

## Build a contract-review assistant that cites its evidence

This workshop teaches supervised fine-tuning of NVIDIA Nemotron 3 models through two alternative Amazon SageMaker AI workflows. Both use ContractNLI: a model reads a non-disclosure agreement, applies a fixed checklist of 17 legal hypotheses, and returns a verdict plus numbered evidence spans for every item. You will follow the data, configuration, model artifact, and evaluation results through their actual service boundaries.

The goal is not simply to obtain fluent text or valid JSON. A reviewer needs a correct decision and an inspectable citation. A model that claims a protection exists when the agreement is silent can be more harmful than one that openly returns an incomplete answer. This educational workflow requires human legal review and is not a substitute for legal advice.

## Learning objectives

By the end of your selected track, you should be able to:

- Transform document-level annotations into the exact supervised record schema consumed by the selected training interface.
- Explain how Low-Rank Adaptation changes a model and why precision, adapter targets, token budgets, and loss boundaries matter.
- Distinguish a SageMaker serverless customization recipe from a Training job running your own TRL script.
- Trace a model from its training input and configuration to either a registry package or a direct S3 artifact.
- Explain the actual deployment resource order and verify the request, output schema, and artifact identity.
- Interpret verdict accuracy, citation F1, schema compliance, and judge coverage without treating a smoke test as a benchmark.
- Identify and clean up independently billed training, evaluation, hosting, Studio, and retained storage resources.

Track 1 implements managed evaluation of registered base/tuned models. Track 2 implements training, direct deployment, held-out endpoint prediction collection with local scoring, and optional Bedrock judging of precomputed responses. Track 2's evaluation switches default to shipped reference results, so running its analysis without enabling live evaluation does not measure your deployed model. Both evaluation lessons distinguish implemented behavior from remaining metric, provenance, and judge limitations.

## Understand the output contract

Each checklist key maps to a verdict and an evidence list:

| Label           | Meaning                                    | Evidence                            |
| --------------- | ------------------------------------------ | ----------------------------------- |
| `Entailment`    | The contract supports the hypothesis       | Supporting clause/span identifiers  |
| `Contradiction` | The contract conflicts with the hypothesis | Conflicting clause/span identifiers |
| `NotMentioned`  | The contract does not address it           | Empty list                          |

For example, the notebook's annotated Navidec contract prohibits disclosure without prior written permission in span `[3]`. The dataset labels employee sharing (`nda-5`) as `Contradiction` with evidence `[3]`. That is a dataset annotation to inspect, not a model result. One clause can justify several checklist decisions, and an exception elsewhere in the contract can change the interpretation.

The full target contains all 17 original keys, which are not consecutively numbered. ContractNLI's document-level split is 423 training, 61 development, and 123 test contracts. Keeping entire documents in one split prevents training on one part of an NDA and testing on another. The test set's 2,091 decisions share 123 documents, so they should not be treated as 2,091 independent contracts.

## Choose one track

| Dimension          | [Track 1: Serverless customization](/02-serverless/)  | [Track 2: Training jobs](/03-training-jobs/)                                  |
| ------------------ | ----------------------------------------------------- | ----------------------------------------------------------------------------- |
| Model              | Nemotron 3 Nano 30B-A3B BF16                          | Nemotron 3 Nano 4B BF16                                                       |
| Model identifier   | JumpStart ID from `config.py`                         | Hugging Face ID from `config.py`                                              |
| Training interface | SageMaker `SFTTrainer`, managed LoRA recipe           | `ModelTrainer` running TRL `SFTTrainer` and PEFT                              |
| Input shape        | String `prompt`/`completion`; test `query`/`response` | Conversational `prompt`/`completion` with template options                    |
| Input handoff      | AI Registry datasets                                  | S3 `train`, `val`, and `config` channels                                      |
| Training output    | Model Package with merged checkpoint prefix           | Merged checkpoint/tokenizer in S3 `model.tar.gz`                              |
| Hosting            | Four-GPU inference component on one `ml.g5.12xlarge`  | Direct production variant on one `ml.g5.xlarge`                               |
| Quality evaluation | Managed custom scorer and LLM judge                   | Endpoint predictions with local scoring; optional Bedrock judge; shipped references by default |

Choose Track 1 to focus on managed recipe configuration, registry lineage, and managed evaluation. Choose Track 2 to inspect training internals, inline YAML, dependencies, model export, and direct serving. The current Training-job recipe uses one `ml.g5.2xlarge`, one epoch, BF16 LoRA rather than QLoRA, and an optional-by-configuration retained warm pool.

These tracks are alternatives, not consecutive labs and not a controlled comparison of model sizes. They differ in model, prompt representation, training recipe, and evaluation implementation. An active-parameter designation describes model computation, not proof of superior task quality or the complete storage requirement for its weights.

::alert[**Use one track per setup.** Both configurations default to `datasets/contractnli-nda-review` in the session bucket, including the same train and validation object keys. Their schemas differ. Running one track's upload cells over the other can invalidate a registered dataset without changing its name. Separate notebook folders do not isolate S3; isolate bucket/prefix and resource naming consistently before attempting both.]{type="warning"}

## How to use the workshop

The notebooks and helpers live in [aws-samples/generative-ai-on-amazon-sagemaker](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/tree/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai). Open them in a SageMaker AI Studio JupyterLab space and execute the selected track from its own directory. The pages explain the transformations, annotated code, checks, and failure modes; the linked notebooks remain the execution source.

| Stage                               | What to retain before proceeding                                                       |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| [Prerequisites](/01-prerequisites/) | Account, Region, role, chosen track, and quota/access checks                           |
| Data preparation                    | Exact schema, split counts, input locations, and prompt/template configuration         |
| Training                            | Job identity, resolved configuration, and package/artifact identity                    |
| Evaluation                          | Actual prediction/reference pairs, scorer definition, execution identity, and coverage |
| Deployment                          | Model, EndpointConfig, Endpoint, and component name where applicable                   |
| [Clean Up](/04-cleanup/)            | Confirmation that unwanted compute and resources are no longer active                  |

Track 2's notebook execution order is `1-prepare-data.ipynb`, `2-fine-tune-llm.ipynb`, `3-deployment.ipynb`, then `4-evaluation.ipynb`. Keep the endpoint running for fresh prediction collection, not merely for reading references or judging saved answers. Hosting cleanup is at the end of notebook 4 and runs independently of the evaluation switches, so do not use Run All without reviewing that final cell. Some source Markdown still describes older multi-GPU settings or historical results; these pages use the executable configuration and distinguish reference results from your own measurements.

## Cost, security, and evidence

Training, evaluation inference, Bedrock judging, endpoint hosting, Studio compute, warm pools, and storage are separate cost categories. Serverless **training** does not make the real-time **endpoint** serverless. Stopping a kernel does not stop a remote job, and a local timeout does not delete an endpoint. No fixed runtime, cost reduction, or quality improvement is promised.

Use approved data and a dedicated workshop environment. Review ContractNLI and model licenses before use; never upload confidential contracts or credentials to a temporary event account. Apply IAM permissions to the actual resources and distinguish role trust from permission to invoke a service. Preserve the data/configuration/artifact lineage so that results can be audited rather than inferred from a familiar resource name.

:button[Start prerequisites]{href="/01-prerequisites/"}
