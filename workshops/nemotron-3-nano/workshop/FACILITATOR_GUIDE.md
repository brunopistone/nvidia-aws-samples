# Facilitator Guide: Fine-tune NVIDIA Nemotron 3 models on Amazon SageMaker AI

## Workshop Overview

The primary journey is **SageMaker Training jobs: 4B**, using ContractNLI to teach supervised fine-tuning, BF16 LoRA, evidence-grounded output, MLflow tracking, merged-artifact deployment, and held-out evaluation. **Optional: Serverless model customization** remains visible as an alternative managed workflow using 30B-A3B. It is not required before the 4B journey and is not a controlled model-size comparison.

Audience: practitioners comfortable with Python notebooks, S3, and basic IAM who want to understand the data, training, hosting, and evaluation boundaries. Participants should leave able to trace an actual job to its artifact, distinguish reference charts from their own results, and verify cleanup. This guide is a presenter outline, not a slide deck or a claim that live training has been validated for a particular event.

## Prerequisites

### Facilitator preparation

- Rehearse the exact published source revision in the intended account/Region before delivery. No event deployment or infrastructure update is part of this documentation change.
- Open `02-smtj-workshop/1-prepare-data.ipynb`, `2-fine-tune-llm.ipynb`, `3-deployment.ipynb`, and `4-evaluation.ipynb` in that order. Check helpers and dependency files from the same revision; do not mix old notebook output with new code.
- Confirm Studio/JupyterLab access, model license/access, approved role and S3 prefix, outbound dependencies, and separate quotas for one `ml.g5.2xlarge` training instance and one `ml.g5.xlarge` endpoint. Warm-pool retention currently defaults to zero. Quota approval is not capacity assurance.
- Leave `MLFLOW_APP_ARN` empty to discover the uniquely matching tagged stack app before either journey's fallback. The stack app takes precedence over a named duplicate; ambiguity requires an explicit ARN. Only if no workshop app matches does the primary journey create `nemotron-contractnli-workshop`, or serverless use native SDK discovery/create unless role/artifact overrides select named creation. Both wait for readiness, create with `AccountDefaultStatus="DISABLED"`, and require no domain or Tracking Server. Invalid explicit ARNs fail without replacement. Rehearse permissions and save the resolved ARN for reuse instead of repeatedly creating apps during the talk.
- Keep `constraints-notebook.txt` and `workshop_mlflow.py` in the parent workshop directory. The successful Linux CPython 3.12-targeted pip resolver dry runs are not a Studio runtime proof. Install the selected notebook requirements, restart, run `%pip check`, and verify imports in the actual environment; validate CUDA training dependencies separately in the DLC.
- Open tabs for Studio **Jobs > Training**, Studio **MLflow**, the SageMaker console **Inference > Endpoints**, and CloudWatch log groups. Record resource names, account, and Region before screen sharing. Avoid displaying credentials or signed MLflow URLs.
- For the optional serverless extension, check its distinct managed-service access and one four-GPU `ml.g5.12xlarge` hosting instance. Isolate S3/resource names before using both workflows; their default dataset keys overlap despite different schemas.
- Prepare an approved completed run/artifact and saved screenshots or exported observations for fallback. Keep a live demo endpoint only for the window needed and assign a cleanup owner. Do not imply a fallback artifact came from participants' just-submitted jobs.

### Participant prerequisites

Use the prerequisite pages for account and Studio setup, not a repeated setup lecture during evaluation. Confirm the primary directory is `02-smtj-workshop/`, even though the sidebar presents it first. Bedrock access is only needed for the optional live judge, not primary preparation/training/deployment or reading reference results.

## Recommended Agenda

The following blocks are **agenda allowances**, not guaranteed AWS runtimes. Fit setup, data preparation, evaluation, breaks, and cleanup around them according to the event length. Start remote work when participants are ready; never promise it will finish during a talk.

| Block                                | Allowance             | Presenter outline and participant activity                                                                                            |
| ------------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Setup and data                       | Event-dependent       | Verify environment, read Navidec spans and three hypotheses, build conversational records, inspect token lengths and split boundaries |
| Training talk while job runs         | About 30 minutes      | Use the five-part outline below; compare actual job state with the appropriate console/log surface                                    |
| Endpoint Q&A/demo while provisioning | About 10 minutes      | Follow artifact to Model/configuration/endpoint, inspect startup logs, explain billing and smoke test versus quality                  |
| Evaluation                           | Event-dependent       | Declare reference/live switches; discuss metric denominators, optional streaming measurements, and judge coverage                     |
| Cleanup and wrap-up                  | Reserve explicit time | Verify endpoint absence, terminal job states, any retained pools, and storage/Studio ownership                                        |

### Training talk outline: approximately 30 minutes

1. **Task and data, 6 minutes.** Display the sourced Navidec excerpt from the homepage/data lesson. Ask which words in span `[3]` contradict employee sharing and why prior-written-permission conditions matter. Explain `choice -> label`, `spans -> evidence`, the 17 nonconsecutive keys, and the 423/61/123 document splits. Gold is a completion during training, not an inference input.
2. **SFT and loss boundaries, 7 minutes.** Use the original SFT figure linked in the lesson. Explain next-token supervision, teacher forcing, prompt/padding masking, and why generation can fail despite low loss. Distinguish a tokenizer limit from characters or serialized bytes; inspect what truncation removes.
3. **LoRA and memory, 7 minutes.** Explain `W_updated = W + (alpha / rank) * B * A`, rank 32/alpha 64, frozen BF16 base weights, and trainable adapter state. This is not QLoRA. Discuss target-module execution paths and why `out_proj` is not blindly included. One GPU, batch one, accumulation four means nominal effective batch four, with `ceil(423 / 4) = 106` illustrative updates, not promised telemetry.
4. **Training progress and MLflow, 7 minutes.** Open the exact job in Studio **Jobs > Training**, follow CloudWatch `/aws/sagemaker/TrainingJobs`, then open the configured app from Studio **MLflow** and experiment `nemotron-3-nano-4b-contractnli-sft`. Identify installation/loading, optimizer updates, validation, merge, and upload stages. Record the run ID and job name. Missing startup metrics do not alone indicate failure; use logs. Validation loss is not verdict accuracy.
5. **Handoff and cost, 3 minutes.** Explain `/opt/ml/model -> model.tar.gz -> DescribeTrainingJob` and why `Completed` is required after export. Show the cost-slide outline below and distinguish training billable seconds from wall time and retained compute.

### Endpoint Q&A/demo outline: approximately 10 minutes

- **3 minutes:** trace the exact completed job's artifact through Model, EndpointConfig, and Endpoint. Confirm `ml.g5.xlarge`, tensor parallel size one, and the served-model alias. There is no inference component in this journey.
- **3 minutes:** inspect **Inference > Endpoints**, current status/failure reason, and CloudWatch `/aws/sagemaker/Endpoints/<name>`. Ask why a longer timeout cannot repair incompatible architecture support or an OOM. Interrupting a local monitor does not delete the endpoint.
- **2 minutes:** if ready, run the smoke test and explain parseability, checklist schema, citation bounds, and legal correctness as different checks. Otherwise show the labeled saved response and move on without inventing a live result.
- **2 minutes:** answer latency/cost questions. Non-streaming full-response latency is not TTFT. Streaming TTFT stops at the first nonempty generated content, not a role/usage event. An idle provisioned endpoint still incurs hosting charges. The bounded performance probe is not the full quality test or a per-request invoice.

## Delivery Tips

### Precomputed demo fallback

If training or endpoint creation does not finish during its agenda block, keep the learning sequence moving. Show an approved prior job's configuration and logs, clearly labeled with its source revision, job/artifact identity, and date. Use notebook 4 with `RUN_ENDPOINT_EVAL = False`, `RUN_JUDGE_EVAL = False`, and `RUN_PERFORMANCE_BENCHMARK = False` to discuss shipped reference results. Those settings avoid these live evaluations but do not remove all AWS setup activity. Review the final cleanup cell before executing it: it is independent of the switches.

The shipped `baselines/` provide saved quality/judge results, not proof of this event's endpoint performance. Use saved performance measurements only if their timing definitions, sample metadata, and provenance are available; otherwise teach the definitions without a numerical latency claim. If using an already-deployed demonstration endpoint, identify it explicitly and keep participant runs separate. Do not overwrite baseline files or delete shared resources.

### Cost slide outline: fill rates before delivery

Title: **Cost follows resource lifetime, not the notebook timer**.

Fill every applicable rate from [SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/) for the event Region and lookup date. Record currency, pricing model, unit, source URL, and any discounts separately. Use SageMaker training/hosting rates, not EC2 rates. No price is prefilled because a Region/date-specific verified rate is required.

| Line item                      | Rate fields to fill                                                       | Formula and measurement basis                                                                                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4B training                    | Region: **_; lookup date: _**; `ml.g5.2xlarge` training rate/hour: \_\_\_ | `sum(job_billable_seconds * instance_count / 3600 * training_rate)`, including billed failed/retried jobs; use job billable-time fields and billing records, not merely epoch-loop time |
| Warm pool                      | Region/date: **_; applicable retention rate/hour: _**                     | `retained_billable_seconds * retained_instance_count / 3600 * rate`; separate from training; current `KEEP_ALIVE_SECONDS = 0` requests no retention, but inspect older jobs             |
| Primary endpoint               | Region/date: **_; `ml.g5.xlarge` real-time hosting rate/hour: _**         | `sum(provisioned_billable_hours_in_interval * instance_count_in_interval * hosting_rate)`; include idle uptime and avoid overlapping interval double counting                           |
| Optional serverless endpoint   | Region/date: **_; `ml.g5.12xlarge` hosting rate/hour: _**                 | Same uptime formula, distinct four-GPU instance; do not replace the primary rate with this one                                                                                          |
| Studio and MLflow              | Region/date and applicable usage rates: \_\_\_                            | Separate notebook/app usage according to current pricing; closing a tab is not stopping compute                                                                                         |
| Storage, logs, transfer        | Region/date, service and units: \_\_\_                                    | Retained S3/EBS/artifacts/logs and transfer remain separate from compute                                                                                                                |
| Optional judging/customization | Region/date, configured model/service and billing units: \_\_\_           | Use applicable Bedrock or serverless customization pricing and actual usage; do not infer these from training-job instance hours                                                        |

For the optional fixed-fleet streaming microbenchmark, `batch_allocation = measured_wall_seconds / 3600 * instance_count * hourly_rate`. Report warmup separately. Do not sum concurrent request durations to estimate fleet cost. `batch_allocation / measured_requests` is a workload allocation, not an AWS per-request invoice. Full endpoint lifetime already contains the benchmark window, so do not add that allocation again to the lifetime total. State excluded charges and do not promise cost savings or break-even volume without a matched workload and measured utilization.

## Troubleshooting

| Symptom                             | Facilitation response                                                                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Wrong model/schema                  | Verify kernel directory, imported `config.py`/helper paths, and S3 objects. Different folders do not isolate fixed upload keys.                                                 |
| Training appears quiet              | Inspect job secondary status and current CloudWatch stream. Installation and loading precede loss metrics; do not submit another job just because a notebook is idle.           |
| Missing MLflow run                  | Check app ARN, readiness, permissions, resolved YAML, current phase, and experiment. Do not create a Tracking Server as a workaround.                                           |
| Dependency or binary import failure | Inspect the first install/import error and exact container/Python/PyTorch/CUDA ABI. Use the supplied pins rather than upgrading everything during delivery.                     |
| Endpoint startup failure            | Capture current log context, Model image/artifact/environment, and failure reason. A local timeout is not cancellation.                                                         |
| Request parses but is incomplete    | Compare exact checklist keys, canonical labels, integer citation IDs and bounds; inspect output budget and finish reason.                                                       |
| Unexpectedly identical scores       | Check reference/live switches and the actual artifact behind a reused endpoint name.                                                                                            |
| Streaming benchmark fails           | Check streaming permission and image support, request byte cap, output cap, and current logs. Do not replace TTFT with a non-streaming duration or estimate tokens from chunks. |
| Cleanup prints `skip`               | Inspect permissions, ownership, and asynchronous deletion state; verify absence rather than accepting a printed skip as success.                                                |

## Resources

- [Primary notebook source](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/tree/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop)
- [Optional serverless source](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/tree/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop)
- [MLflow App setup](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-app-setup.html)
- [SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/)
- [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)
- [ContractNLI](https://stanfordnlp.github.io/contract-nli/)
- [Bedrock Custom Model Import supported architectures](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html#model-customization-import-model-architecture)

NemotronH is not listed in that import architecture list as checked on 2026-10-14. Do not present Custom Model Import as this workshop's deployment alternative. Native model availability in Bedrock is a separate question. The optional judge evaluates precomputed responses without importing Nemotron weights.

## Post-Event Actions

Verify participant-owned endpoints are gone, unwanted jobs are terminal, warm pools are released or expired, and Studio compute is stopped when no longer needed. Review stored artifacts, logs, MLflow history, and S3/EBS retention separately. Follow event-owner procedures for account termination; do not assume termination or a notebook cleanup cell covers every resource, and do not delete shared infrastructure. Collect feedback, preserve approved observations, and update this guide with confirmed delivery findings.

## Additional Notes

Before delivery, fill the presenter-owned details: event Region and pricing date/rates; exact source revision; demo artifact and fallback evidence; who may enable live judging/performance probes; cleanup owner and retention policy; total event length around the 30-minute and 10-minute allowances. These are delivery details to verify, not open decisions about the primary journey.

No deck or recording is bundled. This outline can be converted into slides without reusing the old SFT image that contains unrelated model branding and a confidentiality footer. The lesson's original SVG is model-neutral and has an accessible text description.
