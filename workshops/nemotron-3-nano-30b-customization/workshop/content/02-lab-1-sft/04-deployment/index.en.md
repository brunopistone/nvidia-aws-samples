---
title: "Deployment"
weight: 4
---

We are now ready to deploy the fine-tuned model to a SageMaker real-time endpoint.

The deployment process involves four steps:

1. **Create an Endpoint Configuration** - Define the instance type and routing strategy
2. **Create an Endpoint** - Provision the infrastructure
3. **Create a Model** - Register the fine-tuned model with a DJL LMI serving container
4. **Create an Inference Component** - Attach the model to the endpoint with compute requirements

---

## Option 1: Deploy with Code

::alert[📒 Open the notebook **`lab-1-supervised-fine-tuning/4-deployment.ipynb`**]

## Prerequisites

Retrieve the fine-tuned model from the Model Package Group and extract the S3 path to the merged model weights. The base model comes from `config.py`, and resource names are built from it with a helper that truncates to SageMaker's 63-character limit:

```python
import hashlib

from config import BASE_MODEL_ID

base_model_id = BASE_MODEL_ID

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

We then fetch the latest fine-tuned model package and build the S3 URI to the merged Hugging Face weights:

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
```

## Create Endpoint Configuration

Define the infrastructure for the endpoint:

```python
instance_count = 1
instance_type = "ml.g5.2xlarge"
health_check_timeout = 700

...

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
```

## Create Endpoint

```python
endpoint = Endpoint.create(
    endpoint_name=endpoint_name,
    endpoint_config_name=endpoint_config_name,
)
endpoint.wait_for_status("InService")
```

## Create Model from Model Package

Get the DJL LMI container image URI and configure the serving environment:

```python
CONTAINER_VERSION = "0.36.0-lmi18.0.0-cu128"
inference_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:{CONTAINER_VERSION}"
```

```python
env = {
    "HF_MODEL_ID": "/opt/ml/model",
    "OPTION_TRUST_REMOTE_CODE": "true",
    "OPTION_MODEL_LOADING_TIMEOUT": "3600",
    "OPTION_TENSOR_PARALLEL_DEGREE": "max",
    "SERVING_FAIL_FAST": "true",
    "OPTION_ROLLING_BATCH": "disable",
    "OPTION_ASYNC_MODE": "true",
    "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
    "OPTION_DTYPE": "bf16",
    "OPTION_MAX_MODEL_LEN": json.dumps(1024 * 32),
}
```

:::alert{header="The container version is pinned for a reason" type="warning"}
Two failure modes hit during the build of this lab, both of which look like a broken model rather than a broken image:

- **Architecture support.** The container's bundled Transformers must recognise the model type. `qwen3` works; a newer architecture the image has never seen is rejected with *"Transformers does not recognize this architecture"*.
- **Driver compatibility.** The newest `cu130` images do not start at all on `ml.g5` instances — driver 470 against the ≤570 they expect — failing with `CannotStartContainerError`.

The pin above is verified working for Qwen3 on `ml.g5.2xlarge`. `OPTION_MAX_MODEL_LEN` is set to 32k because the prompt is a whole NDA: the longest test contract is close to 10,000 tokens, so the 4,096-token training cap is not a serving cap.
:::

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

Attach the model to the endpoint with compute resource requirements:

```python
...

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

## Smoke test on a real contract

The LMI container speaks the OpenAI chat schema, and the inference component is selected with its own header rather than inside the body. The request is a plain `invoke_endpoint` call — this is the request an application would make:

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

That check is the point of the cell:

```python
text, usage = ask_endpoint(doc)
pred = parse_verdicts(text)
print(f"valid JSON: {pred is not None} | usage: {usage}")
print(f"answered {len(pred) if pred else 0} of {len(labels)} checklist items")
```

This is a serving check, not a measurement. The base and fine-tuned models are scored **only** by the managed evaluation pipeline in [Evaluation](../03-evaluation), so the lab reports one set of numbers produced one way — an endpoint that answers one contract tells you the deployment works, not how good the model is.

## Clean Up Resources

When you're done, delete the resources to avoid charges. Delete the **inference component first**, then wait before deleting the endpoint — the notebook sleeps 45 seconds between the two, because the endpoint will refuse to delete while a component is still attached:

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
An endpoint bills per instance-hour for as long as it exists, whether or not you send traffic. On the reference run an idle GPU box cost **more than twice all the fine-tuning combined** — this is the one resource in the lab that will keep charging you after you close the notebook. Delete it when done.
:::

## Optional: Deploy to Amazon Bedrock

Instead of a SageMaker real-time endpoint, you can import the merged model into Amazon Bedrock via [Custom Model Import](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html) for fully serverless inference — no endpoints or infrastructure to manage, billed per Custom Model Unit with scale-to-zero.

::alert[📒 Open the notebook **`lab-1-supervised-fine-tuning/4a-deployment-bedrock.ipynb`**]

- **Bedrock Custom Model Import:** Fully serverless, no infrastructure, best when you want simplicity and don't need fine-grained control over hosting. Note that it scales to zero, so the first request after an idle period pays a cold start of roughly 110 seconds — fine for batch contract review, disqualifying for an interactive reviewer unless you keep it warm.
- **SageMaker Endpoint (above):** Dedicated GPU instances with full control over instance type, scaling, and serving config — best for predictable latency or cost optimization at high throughput. Bills per instance-hour for as long as the endpoint exists, whether or not you send traffic.

The notebook submits an import job pointing at the same `merged_model_s3_uri`, waits for completion, then invokes the imported model with the OpenAI Chat Completion schema:

```python
create_job_response = bedrock_client.create_model_import_job(
    jobName=import_job_name,
    importedModelName=imported_model_name,
    roleArn=bedrock_import_role_arn,
    modelDataSource={"s3DataSource": {"s3Uri": merged_model_s3_uri}},
)
```

:::alert{header="AWS Event participants" type="warning"}
Bedrock Custom Model Import jobs (`bedrock:CreateModelImportJob`) **cannot be run in AWS-managed event accounts** (Workshop Studio) due to service policy restrictions. If you are at an AWS-run workshop, use this notebook as a reference only, or run it in your own AWS account.
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
