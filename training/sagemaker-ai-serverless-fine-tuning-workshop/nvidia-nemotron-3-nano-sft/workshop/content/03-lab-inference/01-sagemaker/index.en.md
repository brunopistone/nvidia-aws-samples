---
title: "SageMaker Inference"
weight: 1
---

We are now ready to deploy the fine-tuned model to a SageMaker real-time endpoint using the **vLLM** SageMaker Deep Learning Container.

The deployment process involves four steps:

1. **Create an Endpoint Configuration** - Define the instance type and routing strategy
2. **Create an Endpoint** - Provision the infrastructure
3. **Create a Model** - Register the merged fine-tuned model with the vLLM serving container
4. **Create an Inference Component** - Attach the model to the endpoint with compute requirements

---

## Option 1: Deploy with Code

::alert[📒 Open the notebook **`code/4-deployment.ipynb`**]

## Prerequisites

Retrieve the fine-tuned model from the Model Package Group and build the S3 path to the **merged** model weights. Serverless SFT writes the LoRA adapter merged back into the base model under `checkpoints/hf_merged`, which is what we deploy:

```python
from sagemaker.core import s3
from sagemaker.core.resources import ModelPackage, ModelPackageGroup
from config import BASE_MODEL_ID

base_model_id = BASE_MODEL_ID

response = sm_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)

fine_tuned_model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
model_package = ModelPackage.get(fine_tuned_model_package_arn)

merged_model_s3_uri = (
    s3.s3_path_join(
        model_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri,
        "checkpoints",
        "hf_merged",
    )
    + "/"
)
```

Resource names are derived from `BASE_MODEL_ID` with the same hash-truncation helper used during fine-tuning, so they stay within the SageMaker 63-character limit:

```python
model_name = build_resource_name(base_model_id, "-sft")
endpoint_config_name = build_resource_name(base_model_id, "-sft-config")
endpoint_name = build_resource_name(base_model_id, "-sft-endpoint")
ic_name = build_resource_name(base_model_id, "-sft-ic")
```

## Create Endpoint Configuration

```python
instance_count = 1
instance_type = "ml.g5.12xlarge"
number_of_gpu = 4  # ml.g5.12xlarge has 4x A10G GPUs -> vLLM tensor parallel size
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
            inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
            routing_config={"routing_strategy": "LEAST_OUTSTANDING_REQUESTS"},
        )
    ],
)
```

## Create Endpoint

A SageMaker Endpoint is a fully managed, always-on HTTPS API that hosts your deployed model and serves real-time inference requests.

```python
endpoint = Endpoint.create(
    endpoint_name=endpoint_name, endpoint_config_name=endpoint_config_name
)
endpoint.wait_for_status("InService")
```

## Configure the vLLM container

Get the vLLM Deep Learning Container image URI:

```python
region = sess.boto_region_name

CONTAINER_VERSION = "vllm:0.22.0-gpu-py312-cu130-ubuntu22.04-sagemaker"

inference_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/{CONTAINER_VERSION}"
```

### Custom `nano_v3` reasoning parser

The fine-tuned model emits its chain-of-thought inside `<think>...</think>` tags. A **reasoning parser** tells vLLM's OpenAI-compatible server to strip that block out and return it in a dedicated field, instead of leaving the raw tags inside the answer.

Nemotron 3 uses the same `<think>` convention as DeepSeek-R1, so the parser subclasses the built-in DeepSeek-R1 parser and registers itself under the name `nano_v3`:

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

### vLLM configuration file

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

Both files are uploaded **into the model's S3 prefix**, so SageMaker mounts them alongside the weights at `/opt/ml/model`:

```python
s3_client.upload_file(
    reasoning_parser_file, model_bucket, f"{model_prefix}/{reasoning_parser_file}"
)
s3_client.upload_file(
    vllm_config_file, model_bucket, f"{model_prefix}/{vllm_config_file}"
)
```

### Serving environment variables

The vLLM DLC is configured through `SM_VLLM_*` environment variables, which map to vLLM engine arguments:

```python
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

## Create Model from Model Package

```python
from sagemaker.core.resources import Model
from sagemaker.core.shapes import ContainerDefinition, ModelDataSource, S3ModelDataSource

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

## Create Inference Component

Attach the model to the endpoint, requesting all four GPUs on the instance:

```python
inference_component = InferenceComponent.create(
    inference_component_name=ic_name,
    endpoint_name=endpoint_name,
    variant_name="AllTraffic",
    specification=InferenceComponentSpecification(
        model_name=model_name,
        compute_resource_requirements=InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=10240,
            number_of_accelerator_devices_required=number_of_gpu,
        ),
    ),
    runtime_config=InferenceComponentRuntimeConfig(copy_count=1),
    region=region,
)
inference_component.wait_for_status("InService")
```

:::alert{header="Timing" type="info"}
Provisioning the endpoint and loading a 30B model takes several minutes. `model_data_download_timeout_in_seconds` is set to 700 to allow for the download.
:::

## Test the Endpoint

The system prompt selects the reasoning language — exactly as during training. Change `language` to `Spanish`, `French`, `Italian` or `German` to see the behaviour switch:

```python
system_prompt = """
You are an AI assistant that thinks in {language} but responds in English.

IMPORTANT: Follow this exact format for every response:
1. First, write your reasoning and thoughts inside <think>...</think> tags
2. Then, provide your final answer in English

Always think through the problem in {language}, then translate your conclusion to English for the final response.
"""

prompt = "What is the capital of France, and why did it become the capital?"

system = system_prompt.format(language="Italian")

request_body = {
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ],
    "max_tokens": 4096,
    "temperature": 0.3,
    "top_p": 0.9,
    "stream": True,
}

response = sagemaker_client.invoke_endpoint_with_response_stream(
    EndpointName=endpoint_name,
    InferenceComponentName=ic_name,
    Body=json.dumps(request_body),
    ContentType="application/json",
)
```

Because of the `nano_v3` parser, the streamed deltas carry the chain-of-thought and the answer in **separate fields**. The notebook reads the reasoning field defensively, since its name varies between vLLM versions:

```python
reasoning = delta.get("reasoning") or delta.get("reasoning_content")
content = delta.get("content")
```

The notebook prints them under `--- reasoning ---` and `--- answer ---` headers, so you can verify that the reasoning is in Italian and the final answer is in English.

For a non-streaming request, the same split appears in the response message:

```python
msg = json.loads(resp["Body"].read())["choices"][0]["message"]
print(msg["reasoning"])  # chain-of-thought
print(msg["content"])    # final English answer
```

## Clean Up Resources

Delete the resources in reverse order of creation when you're done:

```python
from sagemaker.core.resources import (
    Endpoint,
    EndpointConfig,
    InferenceComponent,
    Model,
)

InferenceComponent.get(inference_component_name=ic_name).delete()
Model.get(model_name=model_name).delete()
Endpoint.get(endpoint_name=endpoint_name).delete()
EndpointConfig.get(endpoint_config_name=endpoint_config_name).delete()
```

:::alert{header="Important" type="warning"}
A running endpoint is billed for as long as it exists, whether or not you send requests to it. Delete it as soon as you finish the lab. See [Clean Up](/04-cleanup/).
:::

---

## Option 2: Deploy with UI

You can also deploy your fine-tuned model directly from the SageMaker Studio UI.

From the custom model view, choose **Deploy** to launch the deployment wizard, where you can:

- Select the instance type and count
- Configure the endpoint name
- Set environment variables for the serving container

:::alert{header="Note" type="info"}
This lab uses the SDK approach because the deployment needs a custom reasoning parser plugin uploaded next to the model weights — something the wizard does not cover.
:::
