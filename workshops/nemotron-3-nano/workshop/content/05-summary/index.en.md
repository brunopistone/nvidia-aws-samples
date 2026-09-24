---
title: "Summary"
weight: 50
---

## What this workshop teaches

You explored supervised adaptation of NVIDIA Nemotron 3 models to a structured document-review task: apply a fixed 17-item checklist to an NDA and return verdicts with evidence span citations. The same business requirement led to two different engineering workflows. Their training interfaces, data schemas, artifact handoffs, and serving resources should not be treated as interchangeable.

| Track                                       | Implemented path                                                                                                              | Evidence required before claiming completion                                                                                 |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [Serverless customization](/02-serverless/) | Registered datasets, managed 30B-A3B LoRA customization, registry-based evaluation, merged-prefix inference-component hosting | Identified completed job/package, evaluation execution and coverage, and verified serving resources if deployed              |
| [Training jobs](/03-training-jobs/)         | Conversational S3 inputs, one-epoch 4B BF16 LoRA through TRL, merged tarball, direct vLLM endpoint                            | Identified completed job/artifact, smoke test, and explicit distinction between fresh endpoint scores and shipped references |

If you stopped before a stage, describe the stage you actually completed. A submitted job is not a completed job, a completed job is not an evaluated model, and a parseable smoke-test response is not a measured quality improvement.

## Concepts to carry forward

### Data contracts extend beyond file extensions

Both tracks use JSONL, but their records differ. Serverless train/validation examples contain string-valued `prompt` and `completion`; managed test examples use `query` and `response`. Training-job records contain message arrays and a per-record template-options object. The same file extension and task do not make these schemas interchangeable.

The source annotations already contain expert judgments. Data preparation renames `choice` to `label` and `spans` to `evidence`, preserving the dataset's checklist keys and document-level split. The numbered spans are identifiers tied to the original contract. Renumbering or splitting clauses without preserving that relationship would corrupt the citation target.

### Adaptation and execution settings are different decisions

LoRA learns low-rank updates while freezing base weights. BF16 describes the chosen precision path; it does not mean every internal tensor must have that dtype. QLoRA requires a quantized base-loading path that Track 2 explicitly disables. Adapter target selection must follow the actual model computation path, not another architecture's naming conventions.

Batch size, accumulation, number of devices, epochs, token caps, and loss masks together determine the training process. Track 2's current one-GPU recipe has nominal effective batch four, not the four-GPU calculation retained in older notebook prose. Its early-stopping callback does not turn a one-epoch run into a multi-epoch search. Actual logs, not a copied time estimate, establish what ran.

### An artifact handoff is an interface

The serverless package leads to an uncompressed merged-checkpoint prefix. The Training job exposes a gzipped S3 artifact through `DescribeTrainingJob`. A hosting Model binds an image, artifact source, environment, and execution role; it is not the same object as a Model Package. Adding files beside a tarball does not place them inside it.

Track 1 creates endpoint infrastructure and then attaches a Model through an inference component. Track 2 creates the Model first and names it directly in the production variant. Those resource orders reflect different interfaces, not stylistic preferences. Stable resource names and get-first cells require inspection to avoid reusing an old image or checkpoint silently.

### Quality has multiple denominators

Label correctness, contradiction detection, citation overlap, parser success, strict schema compliance, and judge coverage answer different questions. Managed `contradiction_correct` is not the notebook's one-vs-rest `contradiction_f1`. Mean per-contract evidence F1 is not the same as F1 after pooling all citation counts. Fractions and percentages must be normalized before plotting together.

A tolerant JSON parser can accept commentary, ignore extra keys, or normalize citation representations. The helper's evidence score excludes gold-empty items, which leaves some unsupported-citation behavior to separate validation. Semantic judging can identify plausible alternate clauses, but its judgments need explanations, coverage accounting, and human review rather than blind acceptance.

## Honest scope of Track 2 evaluation

The [evaluation lesson](/03-training-jobs/04-evaluation/) follows the implemented `4-evaluation.ipynb` after `3-deployment.ipynb`. It reloads raw ContractNLI test documents, reconstructs the deployed endpoint's name, and can collect predictions and score them locally with `contractnli_scorer.py`. It needs neither an AI Registry test dataset nor a Model Package. Optional Bedrock judging submits precomputed responses from the shipped base model and either shipped or fresh tuned predictions.

Both evaluation switches default to `False`. With those defaults, the statistical and judge tables describe shipped reference results, not your current endpoint. Enable `RUN_ENDPOINT_EVAL` for fresh tuned predictions and separately enable `RUN_JUDGE_EVAL` if you want new judgments; otherwise a fresh statistical result can appear beside a historical judge result. The untuned and frontier statistical baselines are always loaded from supplied files. Record the exact switches, artifact identity, evaluated population, and provenance before claiming a comparison.

Implementation is not proof of a successful live run or production readiness. `contradiction_correct` is recall-like rather than the notebook's advertised F1, parser success does not enforce the full schema, and the current judge prompt omits the system checklist while its carve-out rubric contains conflicting wording. The evaluation lesson explains these limitations in detail. Preserve actual predictions and raw judge outputs, review coverage, and deliberately run notebook 4's hosting cleanup before the broader [Clean Up](/04-cleanup/) checklist.

## Assemble an experiment record

| Record                | Minimum useful contents                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| Input provenance      | Source version, data split, S3 locations or local test file, schema, and prompt/template configuration  |
| Training provenance   | Exact model ID, expanded/resolved configuration, dependencies/image, job identity, and terminal state   |
| Artifact provenance   | Package ARN or `ModelArtifacts.S3ModelArtifacts`, with the producing job                                |
| Evaluation provenance | Generated/reference text, scorer version, aggregation/scale, execution identity, failures, and coverage |
| Serving provenance    | Image, environment, artifact source, resource names, request shape, and smoke-test outcome              |
| Cleanup record        | Which resources were removed, retained, or still require follow-up                                      |

This record makes a result defensible. A chart labeled “fine-tuned model” without an artifact identity and scored population is not enough to reproduce or audit the claim.

## Plan the next controlled experiment

For your own task, begin with approved documents, explicit labels/output schema, and a held-out document set. Establish a base-model baseline under the same prompt and decoding settings before changing training parameters. Vary one meaningful factor at a time, use validation data for iteration, and preserve the final test set for a clearly identified comparison.

Choose failure checks based on application consequences. A vendor-security checklist, claims review, or compliance workflow may share the verdict-plus-evidence pattern, but it needs its own labels, review process, and acceptance criteria. Do not infer production readiness, comparative model superiority, or cost savings from active parameter counts or a single successful request.

For cost analysis, measure the actual workload and distinguish training, evaluation inference, judging, serving utilization, Studio time, warm pools, and storage. No fixed runtime or cost-per-contract advantage is established by this documentation.

## References

- [Workshop notebooks and helpers](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/tree/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai)
- [SageMaker AI model customization](https://aws.amazon.com/sagemaker/ai/model-customization/)
- [Customize a model in SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customization.html)
- [Train a model with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html)
- [NVIDIA Nemotron models](https://developer.nvidia.com/nemotron)
- [ContractNLI dataset and task](https://stanfordnlp.github.io/contract-nli/)

::alert[Before leaving, complete [Clean Up](/04-cleanup/). Endpoints, Studio compute, retained warm pools, active evaluation jobs, registry metadata, and stored artifacts require separate attention.]{type="warning"}

At an instructor-led event, share feedback about the data, training, evaluation, and deployment boundaries that were most difficult to understand. That feedback is more actionable than an unqualified claim that a model “worked.”
