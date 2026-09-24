---
title: "Track 1: Serverless model customization"
weight: 20
---

## Adapt Nemotron 3 Nano 30B-A3B with a managed recipe

This track uses SageMaker AI serverless model customization to fine-tune a model for ContractNLI's 17-item NDA checklist. You supply labeled examples, a supported model, the customization technique, and selected hyperparameters. SageMaker resolves the managed training recipe. The lessons then evaluate the registered result and separately deploy a merged checkpoint to a real-time endpoint.

The executable [config.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/config.py) selects `huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16`. This is the model identity to record, even when older notebook prose discusses another model size. The active-parameter designation does not eliminate the storage needed for all weights or establish task quality by itself.

:image[Supervised fine-tuning concept]{src="/static/images/lab-1-sft/sft_concept.jpeg" height=384}

## What you will build

```text
ContractNLI documents and expert annotations
    -> string prompt/completion JSONL, plus query/response test JSONL
    -> S3 objects and AI Registry datasets
    -> SageMaker SFTTrainer with TrainingType.LORA
    -> Model Package in a shared package group
       -> managed base/tuned evaluation pipelines
       -> merged checkpoint prefix plus serving configuration
          -> EndpointConfig -> Endpoint -> Model -> InferenceComponent
```

The package connects training, evaluation, and deployment, but each operation remains distinct. A successful training job does not establish quality. A registered package does not provision your application endpoint. An endpoint can become ready before the inference component loads its model. Each lesson defines a checkpoint for that boundary.

## Learning goals and lesson order

Work in `generative-ai-on-amazon-sagemaker/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/` after completing [Prerequisites](/01-prerequisites/).

| Lesson | Existing notebook | What you should be able to explain |
| --- | --- | --- |
| [Data preparation](/02-serverless/01-data-preparation/) | `1-prepare-data.ipynb` | How raw `choice`/`spans` annotations become `label`/`evidence` and why test uses a different outer schema |
| [Fine-tuning](/02-serverless/02-fine-tuning/) | `2-fine-tune-llm.ipynb` | How dataset objects, LoRA overrides, role, and package group configure the managed trainer |
| [Evaluation](/02-serverless/03-evaluation/) | `3-evaluation.ipynb` | How verdict/citation metrics differ and how to audit managed output provenance and judge coverage |
| [Deployment](/02-serverless/04-deployment/) | `4-deployment.ipynb` | How a merged prefix, parser configuration, Model, and component produce an invokable endpoint |

## The task and the prompt boundary

Every record presents the full contract and the fixed checklist. The target is strict JSON containing one object per hypothesis with a verdict and cited span identifiers. `NotMentioned` means silence, not uncertainty, and requires an empty evidence list. The provided document-level split keeps the same agreement out of multiple training/validation/test partitions.

Training and managed test data use `C.build_prompt`, a single string with contract before checklist and a final `/no_think` instruction. The serving notebook uses `C.build_messages`, placing the checklist in a system turn before the contract. That shared helper module reduces duplicated wording but does not make the two prompts byte-identical. A serving smoke test validates one request; the quality effect of prompt changes requires a matched evaluation.

## What is managed and what remains your responsibility

| Managed workflow supplies | You must verify |
| --- | --- |
| Recipe resolution and training infrastructure selection | Model/technique availability, licenses, data schema, and supported overrides |
| Dataset and model registry integration | Actual dataset versions, object contents, package lineage, and intended run selection |
| Evaluation orchestration | Scorer correctness, generation parameters, result-file provenance, failures, and coverage |
| Endpoint lifecycle APIs | Image/artifact compatibility, component resources, request format, and cleanup |

The notebook sets ten epochs, global batch size 64, learning rate `0.0001`, LoRA rank 32, and alpha 64. Those are configuration values, not promises about retained dataset counts or runtime. Its hard-coded sequence-filter arithmetic is not current-run telemetry. Inspect resolved defaults and logs before reporting optimizer steps or preprocessing effects.

## Readiness and resource boundaries

Review [account setup](/01-prerequisites/2-account/) for customization, AI Registry, Model Package, evaluation, S3, role-passing, and Bedrock permissions. The endpoint uses one `ml.g5.12xlarge`; the component allocates four accelerators, matching vLLM tensor parallel size four. The minimum component memory request is host memory, not a GPU cache setting.

::alert[Both tracks default to the same `datasets/contractnli-nda-review` S3 prefix, but Track 2 uploads conversational arrays. Choose one track per setup or isolate all dependent configuration before uploading. Registry version names do not preserve old file contents when fixed S3 keys are overwritten.]{type="warning"}

Before leaving each lesson, save the relevant identifiers: dataset name/version and URI, training-job name, Model Package ARN, evaluation execution ARN, and all hosting resource names. The notebook's newest-resource lookups are conveniences, not experiment isolation. Real-time hosting is provisioned compute even though training is serverless; complete [Clean Up](/04-cleanup/) when finished.

Continue to [Data preparation](/02-serverless/01-data-preparation/).
