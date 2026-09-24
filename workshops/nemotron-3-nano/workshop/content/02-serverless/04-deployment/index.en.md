---
title: "Deployment"
weight: 4
---

## What you will learn

You will resolve the customized model's merged weights from Model Registry, package serving configuration with an uncompressed model prefix, create the endpoint and inference component in the notebook's actual order, invoke the model with the correct routing information, and distinguish a successful serving check from a quality evaluation.

::alert[Open [01-serverless-workshop/4-deployment.ipynb](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tune-nvidia-nemotron-3-sagemaker-ai/01-serverless-workshop/4-deployment.ipynb). This lesson deploys Nemotron 3 Nano 30B-A3B to a SageMaker real-time endpoint, not to Bedrock Custom Model Import. Serverless customization does not make this hosting infrastructure serverless.]{type="info"}

## 1. Identify the model artifact

Run the session setup and package-group naming cells. The package lookup selects the newest package in the group, so verify that its lineage matches the completed training job and evaluation you intend to deploy. Reusing a package group across experiments is normal; assuming the newest version is the approved version is not a deployment policy.

```python
from sagemaker.core import s3
from sagemaker.core.resources import ModelPackage

resp = sm_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    SortBy="CreationTime", SortOrder="Descending", MaxResults=1,
)
assert resp["ModelPackageSummaryList"], "no model packages found - run notebook 2 first"
model_package = ModelPackage.get(resp["ModelPackageSummaryList"][0]["ModelPackageArn"])
merged_model_s3_uri = s3.s3_path_join(
    model_package.inference_specification.containers[0]
    .model_data_source.s3_data_source.s3_uri,
    "checkpoints", "hf_merged",
) + "/"
print(merged_model_s3_uri)
```

The expected output shape is an S3 prefix ending in `checkpoints/hf_merged/`. It contains the base weights with the LoRA update applied, plus the configuration/tokenizer artifacts needed for loading. An adapter alone is not a standalone merged model. Confirm this prefix actually exists and contains the intended checkpoint; the path construction assumes the managed recipe's output layout.

This track uses an uncompressed `S3Prefix`. Track 2 instead supplies one gzipped `model.tar.gz` object. Those data sources are not interchangeable: adding files beside an archive does not insert them inside the archive.

## 2. Add the serving parser and configuration

The notebook creates `nano_v3_reasoning_parser.py` locally, writes the following plugin, and uploads it into the merged-model prefix. The plugin registers `nano_v3` with vLLM and extends the DeepSeek-R1 reasoning parser.

```python
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

@ReasoningParserManager.register_module("nano_v3")
class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
    def extract_reasoning(self, model_output, request):
        reasoning_content, final_content = super().extract_reasoning(model_output, request)
        if (
            hasattr(request, "chat_template_kwargs")
            and request.chat_template_kwargs
            and request.chat_template_kwargs.get("enable_thinking") is False
            and final_content is None
        ):
            reasoning_content, final_content = final_content, reasoning_content
        return reasoning_content, final_content
```

This is **container-side plugin code**, shown as the contents of the file the notebook writes, not a cell to import vLLM into your notebook kernel. It changes how reasoning and final content are separated when the specified request condition holds. It does not grade citations, validate a 17-item checklist, or force the model to emit JSON. The notebook's smoke-test request does not explicitly set `chat_template_kwargs`, so do not assume the conditional branch is exercised on every request.

The accompanying `vllm_config.yaml` contains:

```yaml
enable_auto_tool_choice: true
tool_call_parser: qwen3_coder
reasoning_parser: nano_v3
reasoning_parser_plugin: /opt/ml/model/nano_v3_reasoning_parser.py
```

The task itself does not call tools; these are serving-container settings. The notebook parses `merged_model_s3_uri` to obtain its bucket and prefix, uploads both files there, then removes the temporary local copies. Review the destination before upload because this modifies the artifact prefix used by the package. Preserve a deployment manifest or separate artifact version if you need immutable deployments.

## 3. Align infrastructure and model-parallel settings

Run the resource-name helper. It derives `model_name`, `endpoint_config_name`, `endpoint_name`, and `ic_name` from the configured model ID, truncating long names with a hash. Save all four names for cleanup. The configured hardware is one `ml.g5.12xlarge`, and `number_of_gpu = 4` is reused for tensor parallelism and component allocation.

| Setting                  | Notebook value | Interpretation                                                    |
| ------------------------ | -------------- | ----------------------------------------------------------------- |
| Instance count           | `1`            | One provisioned endpoint instance                                 |
| Tensor parallel size     | `4`            | Split serving work across the allocated GPU devices               |
| Maximum model length     | `32768`        | Serving context budget, not the SFT training cap                  |
| Maximum sequences        | `16`           | Configured sequence concurrency limit, not a throughput guarantee |
| GPU memory utilization   | `0.9`          | vLLM memory-planning setting, not a percentage of host RAM        |
| Component minimum memory | `1024 * 96` MB | Host-memory reservation for the component                         |
| Component accelerators   | `4`            | Must agree with the GPUs expected by vLLM                         |

All model weights must be available to the serving runtime, not just the subset active for one token. The 30B-A3B naming is not a four-billion-parameter storage budget or evidence that it is better than the 4B model on this task. Host RAM, GPU weights, attention caches, other model state, and temporary loading buffers are distinct capacity concerns.

## 4. Create the endpoint infrastructure first

This notebook uses inference-component hosting: the production variant defines infrastructure without naming a model. Its actual resource order is **EndpointConfig, Endpoint, Model, InferenceComponent** after artifact preparation. Do not replace that order with Track 2's direct-model sequence.

The following excerpt is the creation branch of the notebook's get-first cell:

```python
from sagemaker.core.resources import Endpoint, EndpointConfig
from sagemaker.core.shapes import ProductionVariant

instance_type = "ml.g5.12xlarge"
number_of_gpu = 4
health_check_timeout = 700

EndpointConfig.create(
    endpoint_config_name=endpoint_config_name,
    execution_role_arn=role,
    production_variants=[ProductionVariant(
        variant_name="AllTraffic",
        instance_type=instance_type,
        initial_instance_count=1,
        model_data_download_timeout_in_seconds=health_check_timeout,
        routing_config={"routing_strategy": "LEAST_OUTSTANDING_REQUESTS"},
    )],
)
```

Despite the variable name `health_check_timeout`, this value is assigned to the model-data **download** timeout in this cell, not an explicit container startup-health-check field. Keep parameter names, rather than informal variable labels, in mind when diagnosing a timeout.

```python
endpoint = Endpoint.create(
    endpoint_name=endpoint_name,
    endpoint_config_name=endpoint_config_name,
)
endpoint.wait_for_status("InService")
```

Use the original get-first cells when rerunning; do not execute a creation branch blindly against an existing name. An `InService` endpoint at this stage confirms infrastructure readiness, not that the model has been loaded into an inference component.

## 5. Bind image, environment, and merged checkpoint

The executable image setting is the plain SageMaker vLLM image below. Older notebook references to a different serving stack do not change that setting.

```python
import json

region = sess.boto_region_name
CONTAINER_VERSION = "vllm:0.22.0-gpu-py312-cu130-ubuntu22.04-sagemaker"
inference_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/{CONTAINER_VERSION}"
env = {
    "SM_VLLM_MODEL": "/opt/ml/model",
    "SM_VLLM_CONFIG": "/opt/ml/model/vllm_config.yaml",
    "SM_VLLM_DTYPE": "bfloat16",
    "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.9",
    "SM_VLLM_MAX_MODEL_LEN": json.dumps(1024 * 32),
    "SM_VLLM_MAX_NUM_SEQS": "16",
    "SM_VLLM_ENABLE_CHUNKED_PREFILL": "true",
    "SM_VLLM_KV_CACHE_DTYPE": "auto",
    "SM_VLLM_TENSOR_PARALLEL_SIZE": str(number_of_gpu),
}
```

The values are strings because they are container environment variables. Chunked prefill and context/concurrency limits influence memory planning, but do not guarantee that arbitrary larger workloads fit. Verify image availability and architecture/plugin compatibility in your environment before substituting another tag.

```python
from sagemaker.core.resources import Model
from sagemaker.core.shapes import ContainerDefinition, ModelDataSource, S3ModelDataSource

Model.create(
    model_name=model_name,
    primary_container=ContainerDefinition(
        image=inference_image,
        model_data_source=ModelDataSource(
            s3_data_source=S3ModelDataSource(
                s3_uri=merged_model_s3_uri,
                s3_data_type="S3Prefix",
                compression_type="None",
            )
        ),
        environment=env,
    ),
    execution_role_arn=role,
)
```

This is again the creation branch. The notebook reuses an existing Model if found. Such reuse does not update its image, environment, or artifact URI. Inspect existing resource definitions before concluding that a rerun deployed a new checkpoint.

## 6. Attach the inference component

```python
from sagemaker.core.resources import InferenceComponent
from sagemaker.core.shapes import (
    InferenceComponentComputeResourceRequirements,
    InferenceComponentRuntimeConfig,
    InferenceComponentSpecification,
)

ic = InferenceComponent.create(
    inference_component_name=ic_name,
    endpoint_name=endpoint_name,
    variant_name="AllTraffic",
    specification=InferenceComponentSpecification(
        model_name=model_name,
        compute_resource_requirements=InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=1024 * 96,
            number_of_accelerator_devices_required=number_of_gpu,
        ),
    ),
    runtime_config=InferenceComponentRuntimeConfig(copy_count=1),
    region=region,
)
ic.wait_for_status("InService")
```

The component selects the Model and reserves resources on the endpoint. Its accelerator allocation must match `SM_VLLM_TENSOR_PARALLEL_SIZE`; reserving fewer devices than the runtime expects can prevent initialization. Wait for **both** endpoint and component readiness before sending a request. A failure here should be investigated in the serving logs, not treated as a reason to retrain immediately.

## 7. Invoke one held-out contract

Run the notebook's dataset-loading and runtime-client cells, then `ask_endpoint(doc)`. The invocation routes with both `EndpointName` and `InferenceComponentName`. The shortened excerpt below preserves the notebook's request shape and assumes `smr`, `doc`, and `labels` have been initialized:

```python
body = {
    "model_name": ic_name,
    "messages": [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in C.build_messages(doc, labels)
    ],
    "max_tokens": 4000,
    "temperature": 0.0,
    "stop": ["<|im_end|>"],
    "stream": False,
}
response = smr.invoke_endpoint(
    EndpointName=endpoint_name,
    InferenceComponentName=ic_name,
    ContentType="application/json",
    Body=json.dumps(body),
)
payload = json.loads(response["Body"].read())
text = payload["choices"][0]["message"]["content"]
usage = payload.get("usage", {})
```

Expect a chat-completion payload with final text at `choices[0].message.content` and, when provided, a `usage` object. The notebook prints elapsed time, parseability, and the number of returned checklist entries. These are shapes and checks, not promised performance results. Inspect the raw payload if content is absent or routed into a reasoning field.

Apply an additional local schema checkpoint after the notebook's `parse_verdicts` call:

```python
pred = parse_verdicts(text)
assert isinstance(pred, dict)
assert set(pred) == set(labels)
span_ids = {i for i, _ in C.doc_spans(doc)}
for item in pred.values():
    assert isinstance(item, dict)
    assert item.get("label") in C.LABELS
    assert isinstance(item.get("evidence"), list)
    assert all(type(i) is int and i in span_ids for i in item["evidence"])
    if item["label"] == "NotMentioned":
        assert item["evidence"] == []
```

Seventeen entries with the wrong keys do not pass this check. Correct keys and in-range citations still do not prove legal correctness. Training used `C.build_prompt`, while serving uses a system/user representation with the checklist before the contract. The smoke test verifies the route and output structure on one document, not that this prompt change preserves held-out quality. Use [evaluation](/02-serverless/03-evaluation/) results with their own provenance and separately validate any production serving prompt.

## Troubleshooting and cleanup

| Symptom                             | Investigation                                                                           |
| ----------------------------------- | --------------------------------------------------------------------------------------- |
| Model prefix not found              | Verify selected package and actual `checkpoints/hf_merged/` output                      |
| Endpoint ready but invocation fails | Check inference-component status and routing name                                       |
| Parser import failure               | Confirm both serving files are under the mounted prefix and plugin APIs match the image |
| GPU initialization failure          | Align tensor parallelism with component accelerators and inspect memory logs            |
| Repeated run serves old behavior    | Inspect reused Model, EndpointConfig, and component definitions                         |
| Incomplete JSON                     | Inspect raw response, reasoning settings, context/output budget, and finish behavior    |

Real-time compute remains provisioned without requests. Use the notebook's cleanup cells in dependency order: inference component, endpoint, endpoint configuration, then Model. Its fixed 45-second wait after component deletion is not proof that deletion finished. Verify each resource is gone and investigate skipped operations before leaving. Complete the shared [Clean Up](/04-cleanup/) checklist for jobs, registry entries, stored artifacts, and Studio resources.
