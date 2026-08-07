---
title: "Deployment"
weight: 4
---

::alert[📒 Open the notebook **`lab-2-direct-preference-optimization-DPO/4-dpo-deployment.ipynb`**]

We are now ready to deploy the fine-tuned model to a SageMaker real-time endpoint.

## Prerequisites

Retrieve the fine-tuned model from the Model Package Group and extract the S3 path to the merged model weights:

```python
import hashlib
import random

base_model_id = "meta-textgeneration-llama-3-2-1b-instruct"
MAX_NAME_LENGTH = 63  # SageMaker resource name limit


def build_resource_name(base, suffix, max_length=MAX_NAME_LENGTH):
    candidate = f"{base}{suffix}"
    if len(candidate) <= max_length:
        return candidate
    digest = hashlib.sha1(base.encode()).hexdigest()[:6]
    # reserve room for the suffix, a hyphen separator, and the 6-char hash
    keep = max_length - len(suffix) - len(digest) - 1
    truncated = base[:keep].rstrip("-")
    return f"{truncated}-{digest}{suffix}"


# keep model_name short enough that the longest derived name (-endpoint) fits the limit
model_name = build_resource_name(
    base_model_id, f"-dpo-{random.randint(100, 100000)}", MAX_NAME_LENGTH - len("-endpoint")
)

model_package_group_name = f"{base_model_id}-dpo"

endpoint_config_name = f"{model_name}-config"
endpoint_name = f"{model_name}-endpoint"
ic_name = f"{model_name}-ic"
```

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
    fine_tuned_model_package_arn = response["ModelPackageSummaryList"][0][
        "ModelPackageArn"
    ]
    fine_tuned_model_package_group_arn = ModelPackageGroup.get(
        model_package_group_name
    ).model_package_group_arn

    model_package = ModelPackage.get(fine_tuned_model_package_arn)

    # get the merged model artifact and deploy it
    merged_model_s3_uri = (
        s3.s3_path_join(
            model_package.inference_specification.containers[
                0
            ].model_data_source.s3_data_source.s3_uri,
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

## Create Endpoint Configuration

Define the infrastructure for the endpoint:

```python
import random
from sagemaker.core.resources import Endpoint, EndpointConfig
from sagemaker.core.shapes import ProductionVariant

instance_count = 1
instance_type = "ml.g5.12xlarge"
number_of_gpu = 4
health_check_timeout = 700

print(f"Creating EndpointConfig: {endpoint_config_name}")
endpoint_config=EndpointConfig.create(
    endpoint_config_name=endpoint_config_name,
    execution_role_arn=role,
    production_variants=[
        ProductionVariant(
            variant_name="AllTraffic",
            instance_type=instance_type,
            initial_instance_count=1,
            model_data_download_timeout_in_seconds=health_check_timeout,
            routing_config={"routing_strategy": "LEAST_OUTSTANDING_REQUESTS"}
        )
    ]
)
```

## Create Endpoint

A SageMaker Endpoint is a fully managed, always-on HTTPS API that hosts your deployed model and serves real-time inference requests.

```python
print(f"Creating Endpoint: {endpoint_name}")
endpoint = Endpoint.create(
    endpoint_name=endpoint_name,
    endpoint_config_name=endpoint_config_name
)
endpoint.wait_for_status("InService")
print(f"Endpoint {endpoint_name} is InService")
```

## Create Model from Model Package

Get the inference image URI:

```python
region = sess.boto_region_name
CONTAINER_VERSION = "0.36.0-lmi18.0.0-cu128"

inference_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:{CONTAINER_VERSION}"
```

Configure the inference environment:

```python
lmi_env = {
    "SERVING_FAIL_FAST": "true",
    "OPTION_ASYNC_MODE": "true",
    "OPTION_ROLLING_BATCH": "disable",
    "OPTION_MAX_MODEL_LEN": "16384",
    "OPTION_TENSOR_PARALLEL_DEGREE": "max",
    "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
    "OPTION_TRUST_REMOTE_CODE": "true",
}
```

```python
from sagemaker.core.resources import Model
from sagemaker.core.resources import TrainingJob
from sagemaker.core.shapes import ContainerDefinition, ModelDataSource, S3ModelDataSource

fine_tuned_model = Model.create(
    model_name=model_name,
    primary_container=ContainerDefinition(
        image=inference_image,
        model_data_source=ModelDataSource(
            s3_data_source=S3ModelDataSource(
                s3_uri=merged_model_s3_uri ,
                s3_data_type="S3Prefix",
                compression_type="None"
            )
        ),
        environment=lmi_env
    ),
    execution_role_arn=role
)

pprint(fine_tuned_model)
```

## Create Inference Component

```python


from sagemaker.core.resources import InferenceComponent
from sagemaker.core.shapes import (
    InferenceComponentSpecification,
    InferenceComponentContainerSpecification,
    InferenceComponentComputeResourceRequirements,
    InferenceComponentRuntimeConfig,
)

# Step 3: Create InferenceComponent
inference_component = InferenceComponent.create(
    inference_component_name=ic_name,
    endpoint_name=endpoint_name,
    variant_name="AllTraffic",
    specification=InferenceComponentSpecification(
        model_name=model_name,
        compute_resource_requirements=InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=10240,
            number_of_accelerator_devices_required=1,
        )
    ),
    runtime_config=InferenceComponentRuntimeConfig(
        copy_count=1
    ),
    region=region
)

print(f"InferenceComponent created: {inference_component.inference_component_name}")
print(f"Endpoint ARN: {endpoint.endpoint_arn}")
inference_component.wait_for_status("InService")
print(f"Endpoint {ic_name} is InService")
```

## Test the Endpoint

Once deployed, test the model with a question. The notebook defines an `execute_inference` helper that wraps the prompt in the Llama chat template and invokes the endpoint (streaming by default):

```python
def execute_inference(prompt, endpoint_name, inference_component_name, stream=True):
    sm_rt_client = boto3.client("sagemaker-runtime")

    if stream:
        result = sm_rt_client.invoke_endpoint_with_response_stream(
            EndpointName=endpoint_name,
            InferenceComponentName=inference_component_name,
            CustomAttributes='accept_eula=true',
            Body=json.dumps(
                {
                    "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".format(prompt),
                    "parameters": {"max_new_tokens": 512, "temperature": 0.1, "top_p": 0.9},
                    "stream": stream
                }
            ),
            ContentType="application/json"
        )
        return result['Body']
```

Then invoke it and stream the response:

```python
prompt = "I've always struggled with math - can you explain how fractals work in a way that's easy to understand?"

stream = execute_inference(prompt, endpoint_name, ic_name, stream=True)
print_stream(stream)
```

The model should respond with a human-like response.

## Clean Up Resources

When you're done, delete the resources to avoid charges:

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

:::alert{header="Important" type="warning"}
Remember to clean up your resources after completing the workshop to avoid ongoing charges.
:::

## Option 2: Deploy with UI

You can also deploy your fine-tuned model directly from the SageMaker Studio UI.

From the custom model view, click **Deploy** to launch the deployment wizard where you can:

- Select the instance type and count
- Configure the endpoint name
- Set environment variables for the serving container

:::alert{header="Note" type="info"}
In this workshop, we use the SDK approach for more control over the deployment configuration. The UI is great for quick experiments and prototyping.
:::

---
