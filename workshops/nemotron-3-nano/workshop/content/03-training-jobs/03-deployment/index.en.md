---
title: "Deployment"
weight: 3
---

## What you will learn

You will retrieve the exact artifact from a completed Training job, distinguish a compressed model object from an uncompressed prefix, configure vLLM for a direct SageMaker endpoint, diagnose startup failures from current logs, and verify the output contract on one held-out NDA.

::alert[Open [02-smtj-workshop/3-deployment.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/02-smtj-workshop/3-deployment.ipynb) after training completes. The notebook deploys directly to a production variant with **no inference component**. It leaves the endpoint running for `4-evaluation.ipynb`, which contains hosting cleanup.]{type="info"}

## Why this lesson uses a SageMaker endpoint

The [Bedrock Custom Model Import documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html#model-customization-import-model-architecture), checked on 2026-10-14, lists supported Mistral, Mixtral, Flan, Llama/Mllama, GPTBigCode, Qwen, and GPT-OSS architectures. **NemotronH is not listed as a supported import architecture.** A merged Hugging Face checkpoint alone does not establish import compatibility, so this workshop does not present Custom Model Import as a deployment option for its Nemotron artifact. This is distinct from native model availability in Bedrock and from Bedrock judging precomputed responses.

## 1. Resolve the completed training artifact

Run the session setup and import the track configuration. If possible, set `TRAINING_JOB_NAME` to the exact run you recorded during training. The default discovery cell queries completed jobs whose names **contain** `TRAIN_JOB_PREFIX`, sorts by creation time, and selects the newest returned job. It is not a strict starts-with filter or proof that the selected run used your current configuration.

```python
from config import BASE_MODEL_ID, MODEL_SLUG, TRAIN_JOB_PREFIX

TRAINING_JOB_NAME = None
if TRAINING_JOB_NAME is None:
    jobs = sm_client.list_training_jobs(
        NameContains=TRAIN_JOB_PREFIX,
        StatusEquals="Completed",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )["TrainingJobSummaries"]
    assert jobs, "No matching completed training job"
    TRAINING_JOB_NAME = jobs[0]["TrainingJobName"]

job = sm_client.describe_training_job(TrainingJobName=TRAINING_JOB_NAME)
model_data_url = job["ModelArtifacts"]["S3ModelArtifacts"]
print(TRAINING_JOB_NAME)
print(model_data_url)
```

The artifact is `model.tar.gz`, produced from `/opt/ml/model` after the merge and tokenizer export. There is no Model Package to query. The notebook checks object size as a warning for an adapter-only export, but a large file is not proof that all configuration, tokenizer, and weight files are correct. The directory may also contain trainer artifacts from the configured output path.

The notebook's size check assumes the artifact is in the session's default bucket. If your job used another output bucket, parse the actual URI rather than interpreting a bucket-split error as a missing model. Confirm the selected job's status, input/configuration provenance, and artifact location before provisioning an endpoint.

## 2. Understand the archive boundary

The Model resource uses `s3_data_type="S3Object"` and `compression_type="Gzip"`. SageMaker downloads and unpacks the object into `/opt/ml/model`, where vLLM loads it. A Model resource is a serving definition binding image, artifacts, and role; it is not a Model Registry package.

This track does not upload a custom reasoning-parser plugin or `vllm_config.yaml` beside the archive. An adjacent S3 object would not become a file inside `model.tar.gz`. The executable notebook instead uses container environment settings and per-request `chat_template_kwargs`. If an application needs additional files in the model directory, that is a separate packaging change requiring validation.

## 3. Name and size the serving resources

Run the notebook's `rname` helper to derive the Model, EndpointConfig, and Endpoint names. It uses `MODEL_SLUG`, which removes the Hugging Face organization prefix and normalizes characters, rather than using the slash-containing `BASE_MODEL_ID` as an AWS resource name. The names are stable across reruns; that helps discovery but makes stale-resource checks important.

| Setting                          | Executable value                      | Meaning                                                       |
| -------------------------------- | ------------------------------------- | ------------------------------------------------------------- |
| Instance                         | One instance across four prioritized Instance Pools | `ml.g5.xlarge` preferred; alternatives listed below |
| GPU count / tensor parallel size | `1`                                   | One serving GPU                                               |
| Model dtype                      | `bfloat16`                            | BF16 serving of the merged model                              |
| Model context limit              | `16384`                               | Total serving context budget, separate from training's `8192` |
| Maximum sequences                | `16`                                  | Configured concurrent-sequence limit                          |
| GPU memory utilization           | `0.9`                                 | vLLM GPU memory-planning setting                              |
| Served model name                | `nemotron-contractnli`                | Name the request's `model` field must match                   |
| Instance provisioning timeout | `900` seconds | Total capacity-provisioning allowance across all pools |
| Download timeout                 | `1200` seconds                        | Model download allowance                                      |
| Startup health-check timeout     | `300` seconds with `FAST_FAIL = True` | Container startup allowance, or `1200` when false             |

GPU memory and host RAM are different constraints. The GPU must accommodate weights and runtime state; the host must accommodate download, unpacking, loading, and other processes. Do not assume a fixed SageMaker host-memory overhead or conclude from the instance name alone that startup must succeed or fail. Use current logs and resource observations. Increasing a timeout does not fix an unsupported configuration or an out-of-memory crash.

### Instance Pools

The production variant uses `instance_pools` instead of a fixed `instance_type`, with this exact order and explicit priorities:

| Priority | Instance type |
| --- | --- |
| 1 | `ml.g5.xlarge` |
| 2 | `ml.g6.xlarge` |
| 3 | `ml.g6.2xlarge` |
| 4 | `ml.g6e.xlarge` |

Priority 1 is highest. SageMaker attempts that pool first and falls back to the next priority on insufficient capacity, until the requested count is provisioned, the pools are exhausted, or the total provisioning timeout is reached. This is capacity fallback, not a guarantee of availability, lowest price, or recovery from a container failure. Check hosting quotas for the intended pool types in your Region. See [Deploy to multiple instance types with instance pools](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-heterogeneous.html).

`initial_instance_count=1` requests one instance total, not one per pool. All four candidates have one GPU and use the same Model, BF16 configuration, and tensor parallel size of one. Their GPU and host resources differ, so startup and serving performance can differ. If you later scale out, a variant can contain multiple instance types at once; the pool list is not evidence that all four types are allocated now.

The notebook sets three separate service timeouts explicitly. `variant_instance_provision_timeout_in_seconds=900` maps to `VariantInstanceProvisionTimeoutInSeconds` and bounds capacity provisioning across all pools, not 900 seconds for each pool. `model_data_download_timeout_in_seconds=1200` maps to `ModelDataDownloadTimeoutInSeconds` for model download. `container_startup_health_check_timeout_in_seconds` maps to `ContainerStartupHealthCheckTimeoutInSeconds` and allows 300 seconds with `FAST_FAIL=True`, or 1200 when false, for container startup health checks. These are notebook values, not claims about API defaults or a single end-to-end deployment deadline. The local `quiet_timeout=25` monitor limit is separate from all three.

## 4. Inspect the image and environment together

The current cell constructs a **plain `vllm` image**, despite nearby notebook prose describing a `huggingface-vllm` image and a Transformers-version table. The executable URI is:

```python
region = sess.boto_region_name
VLLM_VERSION = "0.28.0"
CONTAINER_VERSION = f"vllm:{VLLM_VERSION}-gpu-py312-cu130-ubuntu24.04-sagemaker"
inference_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/{CONTAINER_VERSION}"
print(inference_image)
```

The training dependency file pins Transformers `5.15.1`; this image string does not state its bundled Transformers version. Neither a nearby version number nor a shared minor version guarantees compatibility. Check the actual serving environment and the serialized model configuration. The notebook describes a possible `layers_block_type` validation error involving `full_attention` and `linear_attention`; if you see that error, inspect configuration/schema support before attempting another long startup. Do not delete or rewrite architecture fields merely to make validation pass without understanding the model implementation.

```python
import json

SERVED_MODEL_NAME = "nemotron-contractnli"
number_of_gpu = 1
env = {
    "SM_VLLM_MODEL": "/opt/ml/model",
    "SM_VLLM_SERVED_MODEL_NAME": SERVED_MODEL_NAME,
    "SM_VLLM_DTYPE": "bfloat16",
    "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.9",
    "SM_VLLM_MAX_MODEL_LEN": json.dumps(1024 * 16),
    "SM_VLLM_MAX_NUM_SEQS": "16",
    "SM_VLLM_ENABLE_CHUNKED_PREFILL": "true",
    "SM_VLLM_KV_CACHE_DTYPE": "auto",
    "SM_VLLM_TENSOR_PARALLEL_SIZE": str(number_of_gpu),
}
```

These string-valued variables configure the server. `SM_VLLM_SERVED_MODEL_NAME` gives the OpenAI-compatible API a stable model name instead of relying on a filesystem path as the default. Context length and concurrency affect runtime memory requirements; neither is a throughput promise. Preserve the exact environment with the image and artifact URI when diagnosing failures.

## 5. Create Model, then EndpointConfig, then Endpoint

This sequence differs from inference-component hosting. The production variant names the Model directly, so the Model must exist first. The following is the creation branch of the notebook's get-first Model cell:

```python
from sagemaker.core.resources import Model
from sagemaker.core.shapes import ContainerDefinition, ModelDataSource, S3ModelDataSource

Model.create(
    model_name=model_name,
    primary_container=ContainerDefinition(
        image=inference_image,
        model_data_source=ModelDataSource(
            s3_data_source=S3ModelDataSource(
                s3_uri=model_data_url,
                s3_data_type="S3Object",
                compression_type="Gzip",
            )
        ),
        environment=env,
    ),
    execution_role_arn=role,
)
```

Use the original notebook's existing-resource checks on a rerun. A successful `Model.get(model_name)` does not update that Model to the newly selected artifact, image, or environment. Inspect those fields before reusing it, or create a deliberately versioned deployment rather than silently serving an old run.

```python
from sagemaker.core.resources import Endpoint, EndpointConfig
from sagemaker.core.shapes import InstancePool, ProductionVariant

instance_types = [
    "ml.g5.xlarge",
    "ml.g6.xlarge",
    "ml.g6.2xlarge",
    "ml.g6e.xlarge",
]
instance_count = 1
FAST_FAIL = True
health_check_timeout = 300 if FAST_FAIL else 1200
model_download_timeout = 1200

EndpointConfig.create(
    endpoint_config_name=endpoint_config_name,
    production_variants=[ProductionVariant(
        variant_name="AllTraffic",
        model_name=model_name,
        instance_pools=[InstancePool(instance_type=t, priority=i)
                        for i, t in enumerate(instance_types, start=1)],
        variant_instance_provision_timeout_in_seconds=900,
        initial_instance_count=instance_count,
        initial_variant_weight=1.0,
        model_data_download_timeout_in_seconds=model_download_timeout,
        container_startup_health_check_timeout_in_seconds=health_check_timeout,
        inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
        routing_config={"routing_strategy": "LEAST_OUTSTANDING_REQUESTS"},
    )],
)
```

This is the configuration creation branch. Endpoint configurations are immutable. The full notebook first sorts an existing configuration's pools by priority and checks whether that ordered type list and model name match. That is useful but not a complete comparison of timeouts, AMI, environment, or the underlying Model's artifact. When changing settings, verify all reused resources rather than assuming the name is a version identifier.

```python
endpoint = Endpoint.create(
    endpoint_name=endpoint_name,
    endpoint_config_name=endpoint_config_name,
)
```

There is no `InferenceComponent.create` step and no second model attachment to wait for. Once this direct endpoint reaches `InService`, perform the serving check. Do not copy an `InferenceComponentName` argument into requests for this track.

## 6. Monitor actual startup errors

### Use provisioning time to inspect the endpoint

1. Open **SageMaker AI Studio > Deployments > Endpoints**, or the [SageMaker AI console](https://console.aws.amazon.com/sagemaker/) in the same Region under **Inference > Endpoints**. Select the exact `endpoint_name` printed by the notebook. Observe `Creating`, `InService`, or `Failed`; record any failure reason.
2. Open the endpoint configuration and its `AllTraffic` production variant. Confirm the four pool types, priorities 1 through 4, and initial count of one. Inspect the endpoint's allocated pool types and current counts, not just the requested configuration; `DescribeEndpoint` reports the fleet distribution under `ProductionVariants[].InstancePools`. The notebook prints the requested list and readiness, not that actual allocation, so do not treat `ml.g5.xlarge` as selected merely because it is first. Record the allocation for pricing and performance comparisons, then follow the Model name to verify the artifact URI, image, and environment. This is a direct endpoint, so there is no inference component to attach or wait for.
3. Follow the endpoint's log link, or open [CloudWatch](https://console.aws.amazon.com/cloudwatch/) under **Logs > Log groups > `/aws/sagemaker/Endpoints/<endpoint-name>`**. Select current streams and restrict the time window to this creation attempt. Look for weight loading, memory allocation, server startup, or the first concrete error. Old streams can survive endpoint deletion.
4. While waiting, identify which settings control tensor parallelism, serving context length, and startup timeout. Explain why changing the timeout cannot repair an unsupported architecture or OOM. When `InService`, run the one-contract smoke test below; infrastructure readiness alone does not validate JSON or task quality.

**Checkpoint:** retain endpoint/configuration/Model names plus the current artifact URI. The endpoint's **Monitoring** view and CloudWatch metrics help inspect invocations and resource behavior once traffic starts; service latency metrics are not interchangeable with client-measured streaming TTFT.

Run the notebook's `watch_endpoint(endpoint_name, quiet_timeout=25)` helper after its get-or-create endpoint cell. It polls endpoint status and reads error lines from `/aws/sagemaker/Endpoints/<endpoint-name>`. It uses endpoint creation time to avoid replaying unrelated earlier deployment errors and strips process-ID prefixes to reduce duplicate traceback lines.

The helper requires permission to read CloudWatch logs. Its filtered error view is not a complete log archive; open the full current log stream for surrounding context if necessary. Distinguish a model-data download problem, dependency import problem, architecture configuration failure, GPU initialization failure, and an application health-check failure before choosing a fix.

A `quiet_timeout` exception stops the **notebook monitor**, not the remote deployment. The endpoint can still be creating and consuming resources. Inspect its status and delete an unwanted attempt rather than leaving it unattended. Increasing either local monitoring duration or service startup timeout is appropriate only when the logs show healthy progress that needs more time.

## 7. Smoke test the application request

The elapsed time of this non-streaming call includes waiting for the full response. It is **not time to first token (TTFT)**. TTFT requires a streaming request timed to the first generated content, not the first role-only, usage, or reasoning event. Follow the [evaluation performance section](/03-training-jobs/04-evaluation/#performance-measurement-latency-is-not-ttft) for definitions, warmup separation, percentiles, and measurement metadata; one smoke-test duration is not a benchmark.

The notebook loads `test_docs[0]` and uses the same `C.build_messages` and `C.CHAT_TEMPLATE_KWARGS` used during data preparation. It initializes a runtime client with a 300-second read timeout and then sends a non-streaming request. The following excerpt uses those initialized `smr`, `doc`, and `labels` variables:

```python
body = {
    "model": SERVED_MODEL_NAME,
    "messages": [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in C.build_messages(doc, labels)
    ],
    "chat_template_kwargs": C.CHAT_TEMPLATE_KWARGS,
    "max_tokens": 4000,
    "temperature": 0.0,
    "stream": False,
}
response = smr.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(body),
)
payload = json.loads(response["Body"].read())
text = payload["choices"][0]["message"]["content"]
usage = payload.get("usage", {})
```

The request's `model` must match `SM_VLLM_SERVED_MODEL_NAME`. It is not the SageMaker Model resource name, training-job name, or Hugging Face ID. The runtime route is the endpoint name alone. Expected response shape is a chat-completion object with generated text at `choices[0].message.content` and an optional usage object; inspect the actual payload rather than assuming fixed token counts or latency.

The notebook uses the shared scorer's tolerant `parse_answer`, reports covered keys, and checks entry types, canonical labels, and integer evidence. Add this stricter local checkpoint for in-document citation bounds and empty evidence on `NotMentioned`:

```python
from contractnli_scorer import parse_answer

pred, ok = parse_answer(text)
assert ok and isinstance(pred, dict)
assert set(pred) == set(labels)
span_ids = {i for i, _ in C.doc_spans(doc)}
for answer in pred.values():
    assert isinstance(answer, dict)
    assert answer.get("label") in C.LABELS
    assert isinstance(answer.get("evidence"), list)
    assert all(type(i) is int and i in span_ids for i in answer["evidence"])
    if answer["label"] == "NotMentioned":
        assert answer["evidence"] == []
```

This verifies structure, accepted labels, and citation bounds on one document. It does not prove that the cited clauses support the answer. Shared message construction and template flags reduce prompt drift, but library/template changes can still alter rendered tokens. A successful smoke test is neither held-out accuracy nor a before/after comparison.

## Troubleshooting and next step

| Symptom                          | Targeted investigation                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| No completed job found           | Confirm Region, exact job name, terminal state, and `TRAIN_JOB_PREFIX`               |
| Artifact-size check fails        | Check the URI's actual bucket; do not assume the default bucket                      |
| Unsupported layer/config values  | Compare serialized configuration with the actual image's architecture support        |
| Startup timeout                  | Read current container logs before increasing timeouts or changing instance size     |
| vLLM model-name rejection        | Match request `model` to `SERVED_MODEL_NAME`                                         |
| Wrong model behavior after rerun | Inspect reused Model artifact/image/environment and EndpointConfig                   |
| Incomplete answer                | Check raw content, template flag, finish behavior, context length, and output budget |

Continue to [Evaluation](/03-training-jobs/04-evaluation/) using `4-evaluation.ipynb`. Its Setup imports the track's `evaluation_utils.py` as `E` and uses `E.rname` to reconstruct this endpoint's name, with the same served-model alias. `E.ask_endpoint` and `E.score_all` collect fresh held-out predictions when `RUN_ENDPOINT_EVAL = True`. Keep the endpoint running for that collection. Both evaluation switches default to reference-only analysis, which does not measure this deployment; optional Bedrock judging consumes saved responses and does not need hosting after inference finishes.

There are no teardown cells in the current `3-deployment.ipynb`. Hosting deletion is at the end of `4-evaluation.ipynb`, even if you skip evaluation. That cell is not guarded by either evaluation switch, so review it rather than using Run All indiscriminately. When finished, delete endpoint, endpoint configuration, and Model in dependency order, verify asynchronous completion and resolve skipped operations, then complete [Clean Up](/04-cleanup/), including any retained training warm pool.
