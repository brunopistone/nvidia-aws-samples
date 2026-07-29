---
title: "Bedrock Deployment"
weight: 2
---

[Amazon Bedrock Custom Model Import](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html) lets you import fine-tuned weights into Bedrock and serve them **fully serverless** — no endpoints or infrastructure to manage. Bedrock scales to zero after five minutes of inactivity.

::alert[📒 Open the notebook **`code/4a-deployment-bedrock.ipynb`**]

## When to use this vs. a SageMaker endpoint

|                    | **Bedrock Custom Model Import**                                                | **SageMaker real-time endpoint**                                                  |
| ------------------ | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| **Infrastructure** | None to manage                                                                 | You choose the instance type                                                      |
| **Billing**        | Per Custom Model Unit in 5-minute windows, scale-to-zero, plus monthly storage | Per instance-hour while the endpoint exists                                       |
| **Control**        | Limited to what Bedrock exposes                                                | Full control of container, engine args, scaling                                   |
| **Best when**      | You want simplicity and bursty traffic                                         | You need predictable latency, custom serving config, or high sustained throughput |

---

## Before you start — two blockers

:::alert{header="This model architecture is not supported by Custom Model Import" type="warning"}
Custom Model Import only accepts the architectures listed in [Supported architectures](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html): **Mistral, Mixtral, Flan, Llama 2/3/3.1/3.2/3.3, Mllama, GPTBigCode, Qwen2/2.5/3 and GPT-OSS**.

**NVIDIA Nemotron 3 Nano 30B-A3B is a hybrid Mamba-Transformer MoE architecture and is not on that list**, so an import job for the model you fine-tuned in Lab 1 is expected to fail at the architecture-detection step.

Use this section as a **reference pattern**: the exact same flow works for a model you customized with the Lab 1 notebooks on a supported base model (for example a Qwen3 or Llama base model). To serve Nemotron 3, use the [SageMaker real-time endpoint](../01-sagemaker/) path.
:::

:::alert{header="AWS event participants" type="warning"}
`bedrock:CreateModelImportJob` **cannot be run in AWS-managed event accounts** (Workshop Studio) due to service policy restrictions. If you are at an AWS-led workshop, read through this section without running it, and use your own AWS account afterwards.
:::

Other requirements to be aware of:

- Custom Model Import is available in `us-east-1`, `us-east-2`, `us-west-2` and `eu-central-1`
- Weights must be in Hugging Face **safetensors** format (`.safetensors`, `config.json`, `tokenizer.json`, ...)
- LoRA adapters must be **merged into the base weights** — which is exactly what serverless SFT writes to `checkpoints/hf_merged`
- Max context length below 128K; text models under 200 GB
- You must request a quota increase for `Imported models per account` and `Concurrent model import jobs`

---

## Retrieve the fine-tuned model artifacts

The notebook rebuilds the same hash-truncated Model Package Group name used during fine-tuning, then resolves the merged Hugging Face artifacts:

```python
model_package_group_name = build_resource_name(base_model_id, "-sft-mpg")
model_package_version = "1"

model_package_group = ModelPackageGroup.get(model_package_group_name)

fine_tuned_model_package_arn = f"{model_package_group.model_package_group_arn.replace('model-package-group', 'model-package', 1)}/{model_package_version}"
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

## IAM role for Custom Model Import

Bedrock needs an IAM role that trusts `bedrock.amazonaws.com` and can read the S3 bucket holding the model artifacts. Follow [Create a service role for model import](https://docs.aws.amazon.com/bedrock/latest/userguide/model-import-iam-role.html).

## Submit the import job

Bedrock caps both `importedModelName` and `jobName` at 63 characters, so the notebook hash-truncates them and reserves room for a timestamp that keeps the job name unique:

```python
imported_model_name = build_resource_name(base_model_id, "-sft-imported")
import_job_name = (
    build_resource_name(base_model_id, "-sft-import-job", MAX_NAME_LENGTH - len(timestamp) - 1)
    + f"-{timestamp}"
)

create_job_response = bedrock_client.create_model_import_job(
    jobName=import_job_name,
    importedModelName=imported_model_name,
    roleArn=role,
    modelDataSource={"s3DataSource": {"s3Uri": merged_model_s3_uri}},
)

job_arn = create_job_response["jobArn"]
```

## Wait for the job to complete

The import job takes several minutes. The notebook polls until it reaches a terminal state:

```python
while True:
    response = bedrock_client.get_model_import_job(jobIdentifier=job_arn)
    status = response["status"]
    print(f"Import job status: {status}")

    if status == "Completed":
        imported_model_arn = response["importedModelArn"]
        break
    elif status == "Failed":
        print(f"Import failed: {response.get('failureMessage', 'Unknown error')}")
        break
    else:
        time.sleep(60)
```

:::alert{header="If the job fails" type="info"}
Read `failureMessage` in the response. An unsupported architecture (see the blocker above) is reported here rather than at submission time.
:::

## Test the imported model

Imported models are invoked with `invoke_model` using the [OpenAI Chat Completion schema](https://docs.aws.amazon.com/bedrock/latest/userguide/invoke-imported-model.html) — Bedrock detects the request format from the payload structure. The system prompt selects the reasoning language, as in Lab 1:

```python
from botocore.config import Config

# Imported models can return ModelNotReadyException when they have been
# scaled to zero; retries handle the cold start.
retry_config = Config(retries={"total_max_attempts": 10, "mode": "standard"})
bedrock_runtime = boto3.client(
    "bedrock-runtime", region_name=sess.boto_region_name, config=retry_config
)

system = system_prompt.format(language="Italian")

request_body = json.dumps({
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ],
    "max_tokens": 4096,
    "temperature": 0.3,
    "top_p": 0.9,
})

response = bedrock_runtime.invoke_model(
    modelId=imported_model_arn,
    body=request_body,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response["body"].read())
print(response_body["choices"][0]["message"]["content"])
```

:::alert{header="Note" type="info"}
Custom Model Import supports `InvokeModel` and `InvokeModelWithResponseStream`, but **not** batch inference or CloudFormation. Some architectures (Qwen3, for example) also do not support the Converse API.

Unlike the vLLM endpoint in the previous section, there is no reasoning-parser plugin here — the `<think>...</think>` tags arrive inline in `content`, so parse them yourself if you need the reasoning separated.
:::

## Delete the imported model

:::alert{header="Important" type="warning"}
Custom Model Import is billed per [Custom Model Unit (CMU)](https://docs.aws.amazon.com/bedrock/latest/userguide/import-model-calculate-cost.html) in 5-minute windows starting from the first invocation, **plus monthly storage per CMU**. Bedrock scales compute to zero after five minutes of inactivity, but storage keeps billing until you delete the model.
:::

```python
bedrock_client.delete_imported_model(modelIdentifier=imported_model_name)
```
