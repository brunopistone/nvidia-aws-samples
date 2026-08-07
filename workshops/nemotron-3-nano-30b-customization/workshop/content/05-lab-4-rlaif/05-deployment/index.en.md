---
title: "Model Deployment"
weight: 5
---

## Deploy to SageMaker Endpoint

::alert[📒 Open the notebook **`lab-4-reinforcement-learning-from-ai-feedback/5-deployment.ipynb`**]

### Define Endpoint Parameters

The notebook builds SageMaker resource names from the base model ID, hashing and truncating them when needed to stay within the 63-character limit:

```python
import hashlib
import random

base_model_id = "huggingface-llm-qwen2-5-7b-instruct"
MAX_NAME_LENGTH = 63  # SageMaker resource name limit


def build_resource_name(base, suffix, max_length=MAX_NAME_LENGTH):
    candidate = f"{base}{suffix}"
    if len(candidate) <= max_length:
        return candidate
    digest = hashlib.sha1(base.encode()).hexdigest()[:6]
    keep = max_length - len(suffix) - len(digest) - 1
    truncated = base[:keep].rstrip("-")
    return f"{truncated}-{digest}{suffix}"


model_name = build_resource_name(
    base_model_id, f"-rlaif-{random.randint(100, 100000)}", MAX_NAME_LENGTH - len("-endpoint")
)

endpoint_config_name = f"{model_name}-config"
endpoint_name = f"{model_name}-endpoint"
ic_name = f"{model_name}-ic"
```

The notebook then rebuilds the same Model Package Group name that was used during fine-tuning (base model ID plus the `-rlaif` suffix, hashed and truncated to stay under 63 characters):

```python
MAX_MPG_NAME_LENGTH = 63
suffix = "-rlaif"

candidate = f"{base_model_id}{suffix}"
if len(candidate) > MAX_MPG_NAME_LENGTH:
    digest = hashlib.sha1(base_model_id.encode()).hexdigest()[:6]
    keep = MAX_MPG_NAME_LENGTH - len(suffix) - len(digest) - 1
    truncated = base_model_id[:keep].rstrip("-")
    model_package_group_name = f"{truncated}-{digest}{suffix}"
else:
    model_package_group_name = candidate
```

### Get the Fine-Tuned Model S3 Artifact Path

Retrieve the most recent Model Package from the group, then build the S3 URI of the merged model checkpoint:

```python
from sagemaker.core import s3
from sagemaker.core.resources import ModelPackage, ModelPackageGroup

response = sm_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)

if len(response["ModelPackageSummaryList"]) > 0:
    fine_tuned_model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
    fine_tuned_model_package_group_arn = ModelPackageGroup.get(
        model_package_group_name
    ).model_package_group_arn

    model_package = ModelPackage.get(fine_tuned_model_package_arn)

    # get the merged model artifact and deploy it
    merged_model_s3_uri = (
        s3.s3_path_join(
            model_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri,
            "checkpoints",
            "hf_merged",
        )
        + "/"
    )
else:
    fine_tuned_model_package_arn = None
    fine_tuned_model_package_group_arn = None
    merged_model_s3_uri = None

print(f"Model S3 path: {merged_model_s3_uri}")
```

### Create the Endpoint Config

Define the inference configuration and create the endpoint config:

```python
from sagemaker.core.resources import Endpoint, EndpointConfig
from sagemaker.core.shapes import ProductionVariant

instance_count = 1
instance_type = "ml.g5.2xlarge"
health_check_timeout = 700

endpoint_config = EndpointConfig.create(
    endpoint_config_name=endpoint_config_name,
    execution_role_arn=role,
    production_variants=[
        ProductionVariant(
            variant_name="AllTraffic",
            instance_type=instance_type,
            initial_instance_count=instance_count,
            model_data_download_timeout_in_seconds=health_check_timeout,
            routing_config={"routing_strategy": "LEAST_OUTSTANDING_REQUESTS"},
        )
    ],
)
```

### Create the Endpoint

A SageMaker Endpoint is a fully managed, always-on HTTPS API that hosts your deployed model and serves real-time inference requests:

```python
endpoint = Endpoint.create(
    endpoint_name=endpoint_name, endpoint_config_name=endpoint_config_name
)
endpoint.wait_for_status("InService")
print(f"Endpoint {endpoint_name} is InService")
```

### Create the Model from the Merged Artifacts

The model is served with the DJL LMI container. We point `HF_MODEL_ID` at the merged checkpoint downloaded into the container and enable vLLM async serving:

```python
import json

region = sess.boto_region_name
CONTAINER_VERSION = "0.36.0-lmi18.0.0-cu128"

inference_image = (
    f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:{CONTAINER_VERSION}"
)

env = {
    "HF_MODEL_ID": "/opt/ml/model",  # path to where sagemaker stores the model
    "OPTION_TRUST_REMOTE_CODE": "true",
    "OPTION_MODEL_LOADING_TIMEOUT": "3600",
    "OPTION_TENSOR_PARALLEL_DEGREE": "max",
    "SERVING_FAIL_FAST": "true",
    "OPTION_ROLLING_BATCH": "disable",
    "OPTION_ASYNC_MODE": "true",
    "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
    "OPTION_DTYPE": "bf16",
    "OPTION_QUANTIZE": "fp8",
    "OPTION_MAX_MODEL_LEN": json.dumps(1024 * 32)
}

from sagemaker.core.resources import Model
from sagemaker.core.shapes import (
    ContainerDefinition,
    ModelDataSource,
    S3ModelDataSource,
)

fine_tuned_model = Model.create(
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

### Create the Inference Component

The model is attached to the endpoint as an Inference Component, which lets you manage compute allocation independently:

```python
from sagemaker.core.resources import InferenceComponent
from sagemaker.core.shapes import (
    InferenceComponentSpecification,
    InferenceComponentComputeResourceRequirements,
    InferenceComponentRuntimeConfig,
)

inference_component = InferenceComponent.create(
    inference_component_name=ic_name,
    endpoint_name=endpoint_name,
    variant_name="AllTraffic",
    specification=InferenceComponentSpecification(
        model_name=model_name,
        compute_resource_requirements=InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=10240,
            number_of_accelerator_devices_required=1,
        ),
    ),
    runtime_config=InferenceComponentRuntimeConfig(copy_count=1),
    region=region,
)

inference_component.wait_for_status("InService")
print(f"Endpoint {ic_name} is InService")
```

Deployment typically takes 5-10 minutes as SageMaker provisions the compute instance, downloads the model artifacts, loads the model into memory, and runs health checks.

### Test the Endpoint

The endpoint serves streaming responses in an OpenAI-compatible chat format. The notebook includes a `LineIterator` helper and a `parse_streaming_response` function that decodes each streamed chunk into its `(reasoning, content, tool_calls)` parts:

```python
def parse_streaming_response(line_str):
    """Parse a streaming response line and return (reasoning, content, tool_calls)."""
    if not line_str.strip() or line_str.strip() == "data: [DONE]":
        return None, None, None

    if line_str.startswith("data: "):
        line_str = line_str[6:]

    try:
        data = json.loads(line_str)
        if "choices" in data:
            for choice in data["choices"]:
                delta = choice.get("delta", {})
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                content = delta.get("content")
                tool_calls = delta.get("tool_calls")
                if reasoning or content or tool_calls:
                    return reasoning, content, tool_calls
    except json.JSONDecodeError:
        pass

    return None, None, None
```

Send a prompt and stream the response, printing the model's reasoning and answer separately:

```python
prompt = """
What's your honest take on pineapple on pizza?
"""

request_body = {
    "model_name": ic_name,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ],
    "max_tokens": 4096,
    "temperature": 0.3,
    "top_p": 0.9,
    "stop": ["<|im_end|>"],
    "stream": True,
}

response = sagemaker_client.invoke_endpoint_with_response_stream(
    EndpointName=endpoint_name,
    InferenceComponentName=ic_name,
    Body=json.dumps(request_body),
    ContentType="application/json",
)

reasoning_text = ""
generated_text = ""
tool_calls_data = []
in_reasoning = False

for line in LineIterator(response["Body"]):
    if not line:
        continue
    reasoning, content, tool_calls = parse_streaming_response(line.decode("utf-8"))
    if reasoning:
        if not in_reasoning:
            print("--- reasoning ---")
            in_reasoning = True
        reasoning_text += reasoning
        print(reasoning, end="", flush=True)
    if content:
        if in_reasoning:
            print("\n--- answer ---")
            in_reasoning = False
        generated_text += content
        print(content, end="", flush=True)
    if tool_calls:
        if in_reasoning:
            print("\n--- answer ---")
            in_reasoning = False
        tool_calls_data.extend(tool_calls)
        print(f"\n[tool_calls] {tool_calls}", flush=True)
```

:::alert{header="Key Difference" type="success"}
Compared to the base model's formal, mechanical tone, the RLAIF fine-tuned model responds with more natural, conversational language — personal touches, a consistent voice, and fewer rigid list-style structures.
:::

### Clean Up Resources

When you're done testing, delete the endpoint to avoid charges:

```python
from sagemaker.core.resources import InferenceComponent

# Delete inference component
InferenceComponent.get(inference_component_name=ic_name).delete()

from sagemaker.core.resources import Model

# Delete model
Model.get(model_name=model_name).delete()

from sagemaker.core.resources import Endpoint

# Delete endpoint (optional - if you want to remove the endpoint too)
Endpoint.get(endpoint_name=endpoint_name).delete()

from sagemaker.core.resources import EndpointConfig

# Delete endpoint config (optional)
EndpointConfig.get(endpoint_config_name=endpoint_config_name).delete()
```

---

## Congratulations! 🎉

You've successfully completed Lab 4 and learned how to:

✅ Prepare datasets for RLAIF training  
✅ Configure an AI judge with custom reward prompts  
✅ Fine-tune models using serverless RLAIF with GRPO  
✅ Evaluate improvements with LLM-as-a-Judge  
✅ Deploy fine-tuned models to production endpoints

### Next Steps

- Experiment with different reward prompts to optimize for your use case
- Try different base models and compare results
- Scale up training with larger datasets
- Integrate the endpoint into your applications
