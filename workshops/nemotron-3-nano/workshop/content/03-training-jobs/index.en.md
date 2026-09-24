---
title: "Track 2: SageMaker Training jobs"
weight: 30
---

## Own the training recipe and serving handoff

This track adapts NVIDIA Nemotron 3 Nano 4B to the same ContractNLI NDA checklist using a SageMaker Training job. You supply the container, dependencies, source script, input channels, and inline YAML recipe. SageMaker `ModelTrainer` submits the remote job; TRL `SFTTrainer` and PEFT perform the fine-tuning inside it. The recipe uses **BF16 LoRA**, not four-bit QLoRA.

The executable [config.py](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/config.py) selects `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, a Hugging Face model ID rather than a JumpStart ID. It derives `MODEL_SLUG` for legal AWS resource names and `TRAIN_JOB_PREFIX` for job discovery. This training interface is appropriate when you need explicit control over the script and configuration instead of relying on the serverless recipe surface.

## Follow the actual artifact path

```text
ContractNLI documents and annotations
    -> conversational prompt/completion records plus chat_template_kwargs
    -> S3 train and val objects; local ./tmp/test.jsonl
    -> inline args.yaml uploaded as the config channel
    -> ModelTrainer -> scripts/train.py -> TRL SFTTrainer + PEFT
    -> merged model/tokenizer in /opt/ml/model
    -> S3 model.tar.gz recorded on the completed Training job
    -> Model -> EndpointConfig with ModelName -> Endpoint
    -> one-contract serving smoke test
    -> optional fresh predictions from raw test documents -> local statistical scoring
    -> optional Bedrock judge over precomputed responses -> deliberate hosting cleanup
```

There is no dataset-registry import and no Model Package registration in this path. `DescribeTrainingJob` supplies `ModelArtifacts.S3ModelArtifacts`, which deployment passes as a gzipped `S3Object`. The direct production variant names the Model, so it must be created before the endpoint configuration. No inference component is attached or invoked.

## Learning goals and source filenames

Work in `generative-ai-on-amazon-sagemaker/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/` after [Prerequisites](/01-prerequisites/). The current notebook numbers follow artifact dependencies: prepare, train, deploy, then evaluate.

| Lesson                                                                  | Existing file                                                     | Learning checkpoint                                                                                            |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [Data preparation](/03-training-jobs/01-data-preparation/)              | `1-prepare-data.ipynb`                                            | Build message arrays, preserve gold targets, measure token lengths, and retain the local test set              |
| [Fine-tuning](/03-training-jobs/02-fine-tuning/)                        | `2-fine-tune-llm.ipynb`                                           | Read inline YAML, trace the three channels, inspect adapter placement and export, and record the completed job |
| [Deployment](/03-training-jobs/03-deployment/) | `3-deployment.ipynb` | Resolve the artifact, configure vLLM, monitor startup, and validate one contract response |
| [Evaluation](/03-training-jobs/04-evaluation/) | `4-evaluation.ipynb`, plus `contractnli_scorer.py` | Select reference or live scoring, inspect metrics and optional judge coverage, save results, and clean up hosting |

::alert[Evaluation defaults to `RUN_ENDPOINT_EVAL = False` and `RUN_JUDGE_EVAL = False`, loading shipped reference results. Set the first switch to `True` to score your deployed endpoint; optional live Bedrock judging uses the second switch and requires separate permissions. Setup can still access AWS, and the final cleanup cell is not guarded by either switch. Read the [evaluation lesson](/03-training-jobs/04-evaluation/) before running cells or interpreting a chart as your own result.]{type="warning"}

## Verified executable defaults

| Area             | Current code                                                      | Important distinction                                    |
| ---------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| Training compute | One `ml.g5.2xlarge`, `Torchrun()`                                 | One GPU, not the older four-GPU narrative                |
| Optimization     | One epoch, batch one, accumulation four                           | Nominal effective batch four, not sixteen                |
| Adaptation       | Rank 32, alpha 64, dropout 0.05, BF16                             | Base weights are not loaded in four-bit precision        |
| Length and loss  | `max_length: 8192`, completion-only loss, packing off             | Measure full template-rendered records before truncation |
| Configuration    | `args.yaml` written inline in notebook 2                          | No checked-in `scripts/args.yaml` is required            |
| Warm pool        | `KEEP_ALIVE_SECONDS = 1800`                                       | Can retain billed compute after training                 |
| Serving          | One `ml.g5.xlarge`, plain vLLM `0.28.0`, tensor parallel size one | No inference component; image prose elsewhere is stale   |

The training script is reusable and contains branches for other modalities, quantization, and distributed strategies. Their presence is not evidence that this recipe enables them. Inspect the actual arguments and environment when explaining the run. Similarly, `CheckpointConfig` defines a sync directory, but the inline recipe does not redirect the script's checkpoint placement there or enable resumption; the fine-tuning lesson explains the consequence.

## Why the conversational schema matters

`C.build_messages` constructs system/user turns. Each training record adds one assistant turn containing the gold JSON and a `chat_template_kwargs` object with `enable_thinking: false`. TRL renders the model's own template and supervises the completion. The serving request uses the same message builder and flag to reduce prompt drift. The rendered template, tokenizer versions, loss mask, EOS token, and pad token still require verification.

The task is more than output formatting. All 17 labels must be correct, and each cited span must support its verdict. One successful response with 17 entries only proves that one serving request produced a parseable shape. It does not demonstrate improved held-out verdict accuracy or evidence F1.

## Before launching

Check separate training and endpoint quotas, plus warm-pool quota if keeping the configured retention period. The training container needs access to S3 inputs, its image, Python/binary dependencies, and the configured Hugging Face model. Bedrock is not required for the implemented preparation, training, and deployment sequence. Review [account setup](/01-prerequisites/2-account/) for roles, networking, licenses, and costs.

Both tracks write the same default train/validation S3 keys under `datasets/contractnli-nda-review`. Their record schemas differ. A second notebook folder does not isolate those objects; choose one track per setup or isolate bucket/prefix and resource names consistently before using both.

Retain the expanded recipe, source/dependency versions, job name, artifact URI, hosting names, evaluation switch settings, and actual results. Keep hosting active until any fresh predictions have been collected; saved-reference analysis and Bedrock judging do not need a live endpoint. Use the cleanup section of `4-evaluation.ipynb`, then complete [Clean Up](/04-cleanup/) for jobs, warm pools, S3 objects, and Studio.

Continue to [Data preparation](/03-training-jobs/01-data-preparation/).
