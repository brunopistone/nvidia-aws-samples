---
title: "Clean Up"
weight: 40
---

## Cleanup is part of the experiment

Complete this lesson even if training failed, deployment never reached `InService`, or you stopped before evaluation. A notebook kernel, a remote job, a provisioned endpoint, and stored artifacts have independent lifecycles. Closing a browser tab or interrupting a polling cell does not delete the resources it created.

Before deleting anything, save the non-sensitive evidence you need: source/configuration versions, input locations, training-job identity, package/artifact identity, actual evaluation results and coverage, and deployment names. Then inspect each resource to confirm ownership. Never delete an entire default S3 bucket or shared domain simply because the workshop used it.

## 1. Inventory the resources you actually created

| Track                    | Hosting resources                                                | Other resources to review                                                                                                  |
| ------------------------ | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Serverless customization | `ic_name`, `endpoint_name`, `endpoint_config_name`, `model_name` | Training jobs, evaluation pipelines/jobs, AI Registry datasets/evaluator, Model Package versions/group, S3 outputs         |
| Training jobs            | `endpoint_name`, `endpoint_config_name`, `model_name`            | Training job, retained warm pool, optional Bedrock judge jobs and S3 results, S3 inputs/config/model artifacts, local data |

Use the exact names printed by the deployment notebook. A stable name can belong to an earlier experiment that was reused, and a “latest” lookup can select someone else's run in a shared environment. Check the Region as well as the name. Include partially created and failed deployments; a model resource may exist even if endpoint creation failed.

## 2. Delete serverless-track hosting in dependency order

Open the cleanup section of [01-serverless-workshop/4-deployment.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/4-deployment.ipynb). This track attached a Model through an inference component, so remove that attachment first.

The following are the same deletion calls used by the notebook, separated into stages so you can verify asynchronous completion between them. Run only the calls for resources you own and no longer need:

```python
from sagemaker.core.resources import InferenceComponent, Endpoint, EndpointConfig, Model

InferenceComponent.get(ic_name).delete()
```

Wait until the component has disappeared before continuing. The original notebook sleeps for 45 seconds, but a fixed delay is not confirmation of deletion. Inspect the component list/status and resolve an in-use or access error rather than treating a printed “skip” as success.

```python
Endpoint.get(endpoint_name).delete()
```

Verify the endpoint is deleted, then remove its configuration and Model definition:

```python
EndpointConfig.get(endpoint_config_name).delete()
Model.get(model_name).delete()
```

An endpoint can continue to hold compute after a component is removed, so component deletion alone is not the end of hosting cleanup. Conversely, deleting a Model definition does not remove its S3 weights.

## 3. Delete Training-job-track hosting in dependency order

Use the **Clean up** section of [02-smtj-workshop/4-evaluation.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/4-evaluation.ipynb). The current `3-deployment.ipynb` deliberately leaves hosting running and has no teardown cells. This direct endpoint has no inference component. Cleanup is not guarded by `RUN_ENDPOINT_EVAL` or `RUN_JUDGE_EVAL`; even reference-only analysis can reach deletions if you run every cell.

In a fresh evaluation kernel, run its Setup cell to define `MODEL_SLUG`, `rname`, and `ENDPOINT_NAME`, then verify the reconstructed names against your recorded deployment and account/Region. The following staged equivalent uses those evaluation variables rather than relying on variables from the deployment kernel. Run it only for your own resources that are no longer needed:

```python
from sagemaker.core.resources import Endpoint, EndpointConfig, Model

stem = f"{MODEL_SLUG}-contractnli"
Endpoint.get(ENDPOINT_NAME).delete()
```

Wait until endpoint deletion completes, then run:

```python
EndpointConfig.get(rname(stem, "-sft-cfg")).delete()
Model.get(rname(stem, "-sft-m")).delete()
```

The notebook's cleanup loop catches errors and prints skipped operations. Review those messages individually. An endpoint still deleting can keep dependent cleanup from succeeding, while an authorization error requires a permissions fix. Do not repeatedly submit deletion requests without checking the reason.

The `watch_endpoint` helper's local timeout does not stop the remote deployment. If you abandoned a startup attempt, find and delete that endpoint explicitly. Keep Track 2 hosting only until any wanted fresh predictions have been collected and saved. Reading shipped references or running Bedrock judging over precomputed responses does not require retaining an endpoint. If you stop after deployment, still complete hosting cleanup.

## 4. Check active jobs and retained compute

For Track 1, inspect the specific training job and each SageMaker evaluation pipeline execution you launched. Also inspect any corresponding Bedrock evaluation jobs. Let wanted work finish, or stop unwanted active work through the corresponding service and verify a terminal state. Stopping one orchestration layer should not be assumed to have removed every downstream resource; inspect the actual jobs.

For Track 2, inspect both the Training job and its warm-pool state. The notebook configures `KEEP_ALIVE_SECONDS = 1800`, which can retain billed training compute after the job has completed. Release an unwanted retained pool through the supported SageMaker warm-pool controls or verify that retention has expired. Changing a Python variable to zero after submission does not update an existing job or pool.

If you enabled Track 2's `RUN_JUDGE_EVAL`, also inspect both printed Bedrock evaluation-job ARNs, labeled `base` and `tuned`. A failure during the second submission can leave the first job running. Let wanted jobs finish or stop unwanted jobs through the approved service controls, and verify terminal states; interrupting `wait_for_judge` or deleting the endpoint does not cancel them.

For future one-shot runs, set retention to zero before constructing and submitting the trainer if you do not need a pool. The training stopping condition is also not a cleanup instruction for hosting or Studio; those resources remain separate.

## 5. Review registry entries and stored data

Decide what to retain for reproducibility before deleting objects. A registered name, an S3 artifact, a hosting Model, and an evaluation snapshot are different objects. Removing one does not imply removal of the others.

| Track         | Stored or registered items                                                 | Ownership and dependency check                                                                                    |
| ------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Serverless    | `contractnli-nda-review-train`, `-val`, `-test` datasets                   | Verify versions and S3 keys; another experiment may reference the same objects                                    |
| Serverless    | `contractnli-scorer` evaluator                                             | Confirm it is not reused by other evaluations                                                                     |
| Serverless    | Derived package group and its model versions                               | Preserve approved versions; remove intended package versions before a group when required by service dependencies |
| Serverless    | Dataset prefix, training/merged checkpoint prefix, parser/config files     | Serving files were uploaded inside the merged-model prefix                                                        |
| Serverless    | `contractnli-scorer-eval`, `contractnli-judge-eval`, local `./eval_result` | Keep execution provenance with retained metrics; inspect all runs under shared prefixes                           |
| Training jobs | Train/validation files and `config/args.yaml` under `DATA_PREFIX`          | Fixed object keys may have been reused                                                                            |
| Training jobs | Job-specific `output/model.tar.gz` and configured checkpoint storage       | Confirm exact job/output prefix; do not assume the checkpoint sync prefix contains usable resumable checkpoints   |
| Training jobs | Local `./tmp/test.jsonl`, downloaded dataset/tokenizer caches              | Preserve only approved materials needed for later work                                                            |

Track 2 does not create Model Packages or AI Registry datasets in its implemented workflow. Do not delete another track's registry assets. Its optional live judge writes `contractnli-judge-eval/input/{base,tuned}-30.jsonl` with default settings and service outputs under `contractnli-judge-eval/output` in the session bucket. Inspect the recorded job IDs and actual object keys before retaining or removing them, since this prefix is also used elsewhere in the workshop. The evaluation notebook keeps fresh predictions and summary variables in kernel memory rather than automatically saving new local results files; export approved evidence before stopping the kernel. Preserve shipped `baselines/` as source references, not as your own run output. Both tracks share default dataset keys, so verify object ownership and consumers before removing anything.

The notebooks do not implement a comprehensive registry, S3, and CloudWatch teardown. Perform that review separately using approved console or service procedures, scoped to the recorded resources. Retained S3 objects, object versions where enabled, logs, and space storage can remain after compute is gone. Follow your organization's retention requirements rather than deleting audit evidence indiscriminately.

## 6. Stop Studio and review setup infrastructure

Save approved results, stop running notebook kernels, then stop the JupyterLab space. Review persistent space storage separately; stopping compute does not erase saved files. If you created workshop-specific CloudFormation infrastructure, inspect the stack resources and retention requirements before deleting the stack. Do not remove a shared domain, user profile, VPC, role, bucket, or KMS key used by other workloads.

At an instructor-led event, follow the instructor's account-cleanup guidance. Export only permitted non-sensitive material before temporary access expires, never credentials or unrelated account data.

## Final verification

- The intended endpoints are absent, not merely idle or no longer receiving traffic.
- No unwanted Track 1 inference component remains, and dependent Model/EndpointConfig cleanup succeeded.
- No unwanted training or evaluation job is active, and Track 2 warm-pool retention is resolved.
- Retained datasets, packages, artifacts, results, and logs have an explicit owner and retention decision.
- Studio compute is stopped when no longer needed, and shared infrastructure remains intact.
- Any skipped or failed cleanup action has been investigated rather than silently accepted.

Real-time endpoint compute, Studio compute, training/evaluation usage, retained warm pools, and storage are different cost categories. No single “stop notebook” action covers them all. After verifying cleanup, continue to [Summary](/05-summary/).
