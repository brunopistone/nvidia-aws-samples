---
title: "Deployment"
weight: 5
---

We are now ready to deploy the custom-reward fine-tuned model to a SageMaker real-time endpoint.

SageMaker real-time endpoints use an **Inference Component** pattern that decouples the endpoint infrastructure from the model:

1. **Endpoint Configuration** - Define the instance type and routing strategy
2. **Endpoint** - Provision the compute infrastructure
3. **Model** - Register the fine-tuned weights with a DJL LMI + vLLM serving container
4. **Inference Component** - Attach the model to the endpoint with specific resource requirements

---

## Deploy with Code

::alert[📒 Open the notebook **`lab-3a-custom-reward-function-rlvr/5-deployment.ipynb`**]

### Retrieve the fine-tuned model

Locate the latest model package and extract the S3 path to the merged model weights:

```python
from sagemaker.core.resources import ModelPackage, ModelPackageGroup
from sagemaker.core import s3

base_model_id = "huggingface-reasoning-qwen3-06b"
model_package_group_name = f"{base_model_id}-custom-rw-rlvr"
fine_tuned_model_package_arn = None

if fine_tuned_model_package_arn is None:
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    fine_tuned_model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]

model_package = ModelPackage.get(fine_tuned_model_package_arn)

merged_model_s3_uri = s3.s3_path_join(
    model_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri,
    "checkpoints", "hf_merged"
) + "/"
```

### Create Endpoint Configuration and Endpoint

Define the infrastructure (`ml.g5.2xlarge` — a single NVIDIA A10G GPU) and provision the endpoint:

```python
from sagemaker.core.resources import Endpoint, EndpointConfig
from sagemaker.core.shapes import ProductionVariant

endpoint_config = EndpointConfig.create(
    endpoint_config_name=endpoint_config_name,
    execution_role_arn=role,
    production_variants=[
        ProductionVariant(
            variant_name="AllTraffic",
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            model_data_download_timeout_in_seconds=700,
            routing_config={"routing_strategy": "LEAST_OUTSTANDING_REQUESTS"},
        )
    ],
)

endpoint = Endpoint.create(
    endpoint_name=endpoint_name,
    endpoint_config_name=endpoint_config_name,
)
endpoint.wait_for_status("InService")
```

### Create Model from Model Package

Get the DJL LMI container image and configure the vLLM serving environment:

```python
CONTAINER_VERSION = "0.36.0-lmi18.0.0-cu128"
inference_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:{CONTAINER_VERSION}"

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
        environment=lmi_env,
    ),
    execution_role_arn=role,
)
```

### Create Inference Component

Attach the model to the endpoint with compute resource requirements:

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
```

### Test the Endpoint

Once deployed, test the model with a math question using streaming inference:

```python
prompt = "What is 25 * 4 + 10? Show your reasoning step by step."

stream = execute_inference(prompt, endpoint_name, ic_name, stream=True)
print_stream(stream)
```

The model should respond with step-by-step mathematical reasoning.

:::alert{header="Important" type="warning"}
Remember to clean up your resources after completing the workshop to avoid ongoing charges — see the cleanup cells at the end of the notebook.
:::

---
