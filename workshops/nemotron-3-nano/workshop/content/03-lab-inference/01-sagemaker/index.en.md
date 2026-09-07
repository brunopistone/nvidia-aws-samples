---
title: "Deployment"
weight: 1
---

We are now ready to deploy the fine-tuned model to a SageMaker AI real-time endpoint.

Two ways to serve a customized model exist, and they suit different demand shapes. This section deploys the first:

|                | **SageMaker AI real-time endpoint**         | **Bedrock Custom Model Import**                      |
| -------------- | ------------------------------------------- | ---------------------------------------------------- |
| infrastructure | dedicated GPU instance you choose           | fully serverless                                     |
| billing        | per instance-hour, 24/7 while it exists     | per Custom Model Unit in 5-minute active windows     |
| idle cost      | **you keep paying**                         | scales to zero                                       |
| cold start     | none once `InService`                       | tens of seconds after idle                           |
| best for       | steady high throughput                      | spiky or low-volume workloads                        |

Read the right-hand column as the trade you are making by choosing an endpoint: you pay for idle time in exchange for no cold start.

The real-time path involves six steps:

1. **Locate the merged checkpoint** - Read the latest model package from the registry
2. **Add the vLLM configuration** - Upload a serving config and the reasoning parser next to the weights
3. **Create an Endpoint Configuration** - Define the instance type and routing strategy
4. **Create an Endpoint** - Provision the infrastructure
5. **Create a Model** - Register the fine-tuned model with the SageMaker vLLM serving container
6. **Create an Inference Component** - Attach the model to the endpoint with compute requirements

Every create in the notebook is wrapped in a `get`-first `try/except`, so the cells are safe to re-run: an existing resource is reused instead of raising.

---

## Option 1: Deploy with Code

::alert[📒 Open the notebook **`lab-1-supervised-fine-tuning/4-deployment.ipynb`**]

## Prerequisites

Retrieve the fine-tuned model from the Model Package Group. The base model comes from `config.py`, and the group name is rebuilt with the same hash-truncation used by notebooks 2 and 3, so all of them derive the same name for any model id:

```python
import hashlib

from config import BASE_MODEL_ID

base_model_id = BASE_MODEL_ID

MAX_MPG_NAME_LENGTH = 63
suffix = "-contractnli-sft-mpg"

candidate = f"{base_model_id}{suffix}"
if len(candidate) > MAX_MPG_NAME_LENGTH:
    digest = hashlib.sha1(base_model_id.encode()).hexdigest()[:6]
    # reserve room for the suffix, a hyphen separator, and the 6-char hash
    keep = MAX_MPG_NAME_LENGTH - len(suffix) - len(digest) - 1
    truncated = base_model_id[:keep].rstrip("-")
    model_package_group_name = f"{truncated}-{digest}{suffix}"
else:
    model_package_group_name = candidate
```

The newest package in the group is the one to deploy, and what we want from it is the **merged** checkpoint — base weights with the LoRA adapter already applied — not the adapter on its own:

```python
from sagemaker.core import s3
from sagemaker.core.resources import ModelPackage

resp = sm_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    SortBy="CreationTime", SortOrder="Descending", MaxResults=1)
assert resp["ModelPackageSummaryList"], "no model packages found - run notebook 2 first"

model_package = ModelPackage.get(resp["ModelPackageSummaryList"][0]["ModelPackageArn"])

# the *merged* checkpoint (base weights + LoRA applied) is what we deploy
merged_model_s3_uri = s3.s3_path_join(
    model_package.inference_specification.containers[0]
    .model_data_source.s3_data_source.s3_uri,
    "checkpoints", "hf_merged") + "/"
```

The `assert` is deliberate: an empty group means notebook 2 has not finished, and failing here is cheaper than failing 15 minutes into an endpoint creation.

## Add the vLLM configuration to the model artifacts

Nemotron 3 Nano is a reasoning model: it emits its chain of thought inside `<think>...</think>` before the answer. To have the container return that as a separate `reasoning_content` field rather than leaving it inline, vLLM needs a **reasoning parser**. We register a small custom parser named `nano_v3` that inherits the DeepSeek-R1 parser and corrects the one case it gets wrong for this model — when `enable_thinking` is `False`, the base parser puts the answer in the reasoning slot and leaves the content empty:

```python
reasoning_parser_file = "nano_v3_reasoning_parser.py"

nano_v3_reasoning_parser = '''from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


@ReasoningParserManager.register_module("nano_v3")
class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
    def extract_reasoning(self, model_output, request):
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )
        if (
            hasattr(request, "chat_template_kwargs")
            and request.chat_template_kwargs
            and request.chat_template_kwargs.get("enable_thinking") is False
            and final_content is None
        ):
            reasoning_content, final_content = final_content, reasoning_content

        return reasoning_content, final_content
'''

with open(reasoning_parser_file, "w") as f:
    f.write(nano_v3_reasoning_parser)
```

The parser is selected in a YAML config, together with the tool-call parser:

```python
vllm_config_file = "vllm_config.yaml"

vllm_config = """enable_auto_tool_choice: true
tool_call_parser: qwen3_coder
reasoning_parser: nano_v3
reasoning_parser_plugin: /opt/ml/model/nano_v3_reasoning_parser.py
"""

with open(vllm_config_file, "w") as f:
    f.write(vllm_config)
```

Both files are uploaded **into the merged model prefix**, which is what makes them visible to the container: SageMaker mounts that prefix at `/opt/ml/model`, so the paths in the YAML resolve at start-up:

```python
import os
from urllib.parse import urlparse

parsed = urlparse(merged_model_s3_uri)
model_bucket = parsed.netloc
model_prefix = parsed.path.lstrip("/").rstrip("/")

s3_client.upload_file(
    reasoning_parser_file, model_bucket, f"{model_prefix}/{reasoning_parser_file}"
)
s3_client.upload_file(
    vllm_config_file, model_bucket, f"{model_prefix}/{vllm_config_file}"
)

os.remove(reasoning_parser_file)
os.remove(vllm_config_file)
```

:::alert{header="Why the parser ships with the weights" type="info"}
A reasoning parser plugin is loaded from a file path inside the container, so it has to arrive with the model artifacts — there is no way to pass Python code through an environment variable. Uploading it next to the checkpoint keeps the endpoint self-contained: the same S3 prefix carries the weights, the serving config and the parser.

`tool_call_parser: qwen3_coder` is set for the same reason — it is the parser whose format matches the tool calls this model emits.
:::

## Resource names

SageMaker caps resource names at 63 characters, and the base model id is already long, so names are built with a helper that hash-truncates rather than failing:

```python
import hashlib

MAX_NAME = 63


def rname(base, suffix):
    cand = f"{base}{suffix}"
    if len(cand) <= MAX_NAME:
        return cand
    digest = hashlib.sha1(base.encode()).hexdigest()[:6]
    keep = MAX_NAME - len(suffix) - len(digest) - 1
    return f"{base[:keep].rstrip('-')}-{digest}{suffix}"


stem = f"{base_model_id}-contractnli"
model_name = rname(stem, "-sft-m")
endpoint_config_name = rname(stem, "-sft-cfg")
endpoint_name = rname(stem, "-sft-ep")
ic_name = rname(stem, "-sft-ic")
```

## Create Endpoint Configuration

Pick the instance first. This is a 30B MoE model served in bfloat16, so it needs more than one GPU:

```python
instance_count = 1
instance_type = "ml.g5.12xlarge"
number_of_gpu = 4  # ml.g5.12xlarge has 4x A10G GPUs -> vLLM tensor parallel size
health_check_timeout = 700
```

`number_of_gpu` is reused further down as the vLLM tensor parallel size, so the instance choice and the serving configuration cannot drift apart. `health_check_timeout` is generous on purpose: the merged checkpoint is tens of gigabytes, and the download counts against the container's start-up window.

```python
from sagemaker.core.resources import Endpoint, EndpointConfig
from sagemaker.core.shapes import ProductionVariant

try:
    EndpointConfig.get(endpoint_config_name)
    print(f"endpoint config exists: {endpoint_config_name}")
except Exception:
    EndpointConfig.create(
        endpoint_config_name=endpoint_config_name,
        execution_role_arn=role,
        production_variants=[ProductionVariant(
            variant_name="AllTraffic",
            instance_type=instance_type,
            initial_instance_count=1,
            model_data_download_timeout_in_seconds=health_check_timeout,
            routing_config={"routing_strategy": "LEAST_OUTSTANDING_REQUESTS"})])
    print(f"created endpoint config: {endpoint_config_name}")
```

## Create Endpoint

A SageMaker Endpoint is a fully managed, always-on HTTPS API that hosts your deployed model and serves real-time inference requests.

```python
try:
    endpoint = Endpoint.get(endpoint_name)
    print(f"endpoint exists: {endpoint_name}")
except Exception:
    endpoint = Endpoint.create(endpoint_name=endpoint_name,
                               endpoint_config_name=endpoint_config_name)
    print(f"creating endpoint: {endpoint_name}")

endpoint.wait_for_status("InService")
```

The endpoint comes up empty. Because we are using inference components, the model is attached to it afterwards rather than baked into the endpoint configuration.

## Create the Model

The vLLM container is configured entirely through environment variables. Each `SM_VLLM_*` variable maps to the matching `vllm serve` argument, and `SM_VLLM_CONFIG` points at the YAML we uploaded with the weights:

```python
import json

env = {
    "SM_VLLM_MODEL": "/opt/ml/model",  # path where SageMaker mounts the model
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

Three of these are worth reading twice:

- **`SM_VLLM_MAX_MODEL_LEN` is 32k**, not the 4,096-token cap used in training. The prompt here is a whole NDA — the longest test contract is close to 10,000 tokens — so a training cap is not a serving cap.
- **`SM_VLLM_TENSOR_PARALLEL_SIZE` is the GPU count**, sharding the model across all four A10Gs on the instance.
- **`SM_VLLM_ENABLE_CHUNKED_PREFILL` with `SM_VLLM_MAX_NUM_SEQS: 16`** keeps long-prompt prefills from starving each other; with contracts this long, prefill dominates.

The serving container is the SageMaker vLLM Deep Learning Container, pinned by version:

```python
region = sess.boto_region_name
CONTAINER_VERSION = "vllm:0.22.0-gpu-py312-cu130-ubuntu22.04-sagemaker"

inference_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/{CONTAINER_VERSION}"
```

:::alert{header="The container version is pinned for a reason" type="warning"}
The image's bundled vLLM must recognise the model architecture. A container older than the model will reject the checkpoint outright with *"does not recognize this architecture"*, which looks like a broken model rather than a stale image. The `reasoning_parser_plugin` mechanism the config depends on also needs a recent vLLM.

The pin above is the version verified end-to-end for this lab on `ml.g5.12xlarge`. If you change the base model in `config.py`, expect to revisit it.
:::

```python
from sagemaker.core.resources import Model
from sagemaker.core.shapes import (ContainerDefinition, ModelDataSource, S3ModelDataSource)

try:
    Model.get(model_name)
    print(f"model exists: {model_name}")
except Exception:
    Model.create(
        model_name=model_name,
        primary_container=ContainerDefinition(
            image=inference_image,
            model_data_source=ModelDataSource(
                s3_data_source=S3ModelDataSource(
                    s3_uri=merged_model_s3_uri, s3_data_type="S3Prefix",
                    compression_type="None")),
            environment=env),
        execution_role_arn=role)
    print(f"created model: {model_name}")
```

`compression_type="None"` with `s3_data_type="S3Prefix"` means the container reads the checkpoint files straight from the prefix — there is no tarball to build or unpack.

## Create Inference Component

Attach the model to the endpoint with its compute resource requirements:

```python
from sagemaker.core.resources import InferenceComponent
from sagemaker.core.shapes import (InferenceComponentComputeResourceRequirements,
                                   InferenceComponentRuntimeConfig,
                                   InferenceComponentSpecification)

try:
    ic = InferenceComponent.get(ic_name)
    print(f"inference component exists: {ic_name}")
except Exception:
    ic = InferenceComponent.create(
        inference_component_name=ic_name,
        endpoint_name=endpoint_name,
        variant_name="AllTraffic",
        specification=InferenceComponentSpecification(
            model_name=model_name,
            compute_resource_requirements=InferenceComponentComputeResourceRequirements(
                min_memory_required_in_mb=10240,
                number_of_accelerator_devices_required=1)),
        runtime_config=InferenceComponentRuntimeConfig(copy_count=1),
        region=region)
    print(f"creating inference component: {ic_name}")

ic.wait_for_status("InService")

TUNED_MODEL_ID = f"sm:{endpoint_name}/{ic_name}@{region}"
```

The component, not the endpoint, is what actually loads the weights — so this is the wait that takes minutes. `TUNED_MODEL_ID` is the `sm:endpoint/component@region` handle the notebook prints once the model is live.

## Smoke test on a real contract

The vLLM container speaks the OpenAI chat schema, and the inference component is selected with its own header rather than inside the body. The request is a plain `invoke_endpoint` call — this is the request an application would make:

```python
import contractnli as C

C.ensure_dataset("./data")
test_docs, labels = C.load("test")
doc = test_docs[0]

smr = boto3.client("sagemaker-runtime", region_name=region,
                   config=Config(read_timeout=300, retries={"total_max_attempts": 3}))


def ask_endpoint(doc, max_tokens=4000):
    """Invoke the deployed model and return (text, usage)."""
    body = {
        "model_name": ic_name,
        "messages": [{"role": m["role"],
                      "content": [{"type": "text", "text": m["content"]}]}
                     for m in C.build_messages(doc, labels)],
        "max_tokens": max_tokens,
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
    return payload["choices"][0]["message"]["content"], payload.get("usage", {})
```

`temperature` is **0.0**, not a sampling temperature. This is a classification task with a fixed output schema, and there is nothing to gain from variety in the answer.

The turns come from `C.build_messages`, imported rather than rebuilt: pasting a second copy of the prompt into a notebook is how train/serve skew gets introduced.

:::alert{header="The served prompt is not byte-identical to the trained one" type="warning"}
Notebook 1 trains on `C.build_prompt` — a **single string**, contract before checklist. This endpoint needs **turns**, which is what `C.build_messages` returns: the checklist in a system turn, the contract in the user turn. Same instruction, same checklist, same `/no_think`, different order.

That is a deliberate trade rather than an oversight — a chat endpoint wants a system turn, and this instruction is explicit enough to survive the reordering. But it is exactly the kind of difference that silently costs a fine-tune its gains, so the notebook **checks** that the served model still returns all 17 items as parseable JSON instead of assuming it, rather than taking it on trust.

If you would rather serve the trained string verbatim, send a single user turn containing `C.build_prompt(doc, labels)` and nothing else.
:::

That check is the point of the cell. `parse_verdicts` strips any `<think>` block and JSON fence, then parses what is left:

```python
t0 = time.time()
text, usage = ask_endpoint(doc)
elapsed = time.time() - t0

pred = parse_verdicts(text)
ok = pred is not None
print(f"first call {elapsed:.1f}s | valid JSON: {ok} | usage: {usage}")
print(f"answered {len(pred) if ok else 0} of {len(labels)} checklist items")
```

This is a serving check, not a measurement. The base and fine-tuned models are scored **only** by the managed evaluation pipeline in [Evaluation](/02-lab-sft/03-evaluation/), so the lab reports one set of numbers produced one way — an endpoint that answers one contract tells you the deployment works, not how good the model is.

## Clean Up Resources

:::alert{header="An idle endpoint is the most expensive thing in this lab" type="warning"}
An endpoint bills per instance-hour for as long as it exists, whether or not you send traffic. On the reference run, an idle GPU box cost more than **twice** all the fine-tuning combined. Delete it when you are done.
:::

Delete the **inference component first**, then wait before deleting the endpoint — the notebook sleeps 45 seconds between the two, because the endpoint will refuse to delete while a component is still attached:

```python
import time

for label, fn in [
    ("inference component", lambda: InferenceComponent.get(ic_name).delete()),
    ("endpoint", lambda: Endpoint.get(endpoint_name).delete()),
    ("endpoint config", lambda: EndpointConfig.get(endpoint_config_name).delete()),
    ("model", lambda: Model.get(model_name).delete()),
]:
    try:
        fn()
        print(f"deleted {label}")
    except Exception as e:
        print(f"skip {label}: {type(e).__name__}")
    if label == "inference component":
        time.sleep(45)
```

Each delete is guarded, so a resource that was never created — or already removed — is reported and skipped instead of stopping the loop.

## Option 2: Deploy with UI

You can also deploy your fine-tuned model directly from the SageMaker Studio UI.

From the custom model view, click **Deploy** to launch the deployment wizard where you can:

- Select the instance type and count
- Configure the endpoint name
- Set environment variables for the serving container

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach for more control over the deployment configuration. The UI is great for quick experiments and prototyping — but the `nano_v3` reasoning parser above still has to be uploaded alongside the weights, so an endpoint created from the wizard will return the chain of thought inline unless you do that step first.
:::

---
