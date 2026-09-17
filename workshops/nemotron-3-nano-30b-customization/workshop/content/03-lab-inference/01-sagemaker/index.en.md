---
title: "Deployment"
weight: 1
---

We are now ready to deploy the fine-tuned model to a SageMaker AI real-time endpoint.

::alert[Open the notebook **`lab-1-supervised-fine-tuning/4-deployment.ipynb`**]

---

## The four resources

SageMaker's inference stack is split into four separate resources. This separation lets you swap models on a running endpoint, scale copies of a model independently, and reuse endpoint configs across multiple endpoints.

| Class | Resource | What it does |
|---|---|---|
| `Model` | `sagemaker.core.resources.Model` | Registers the merged checkpoint and serving container. Nothing runs yet. |
| `EndpointConfig` | `sagemaker.core.resources.EndpointConfig` | Defines the instance type and routing strategy. |
| `Endpoint` | `sagemaker.core.resources.Endpoint` | The always-on HTTPS API. Comes up empty when using inference components. |
| `InferenceComponent` | `sagemaker.core.resources.InferenceComponent` | Attaches the model to the endpoint and actually loads the weights onto the GPU. |

Every create in the notebook is wrapped in a `get`-first `try/except`. An existing resource is reused instead of raising, so all cells are safe to re-run.

---

## What happens at each step

The notebook walks through six steps in order:

1. **Locate the merged checkpoint.** The training job registered two artifacts: the LoRA adapter on its own, and a merged checkpoint with the adapter already folded into the base weights. We deploy the merged one. A serving engine has to load the full model regardless, so there is no benefit to keeping the adapter separate.
2. **Create the Model.** Registers the S3 path and the LMI container with its environment variables. No weights load yet.
3. **Create the EndpointConfig.** Picks `ml.g5.xlarge` (one A10G GPU) and `LEAST_OUTSTANDING_REQUESTS` routing.
4. **Create the Endpoint.** Provisions the instance. Takes 5-10 minutes. The wait here is for the machine to boot, not for the model.
5. **Create the InferenceComponent.** Downloads the merged checkpoint from S3 and loads it into GPU memory. This wait is longer than the endpoint's.
6. **Smoke test.** Sends one real contract through the endpoint and checks that all 17 verdicts come back as valid JSON.

---

## Key decisions

### The LMI container

The notebook uses the AWS LMI (Large Model Inference) DJL container. Under the hood, LMI runs vLLM when `OPTION_ENTRYPOINT` points to `djl_python.lmi_vllm.vllm_async_service`. The container version is pinned: older images reject Nemotron's architecture outright, and newer `cu130` images fail to start on `ml.g5` hardware. The pin in the notebook is verified working.

### The prompt format

Notebook 1 trains on `C.build_prompt`, a single string with the contract before the checklist. This endpoint expects chat turns, so the smoke test uses `C.build_messages` instead: checklist in the system turn, contract in the user turn. Same instruction, same checklist, different order. The notebook checks that this reordering hasn't broken anything by asserting all 17 items come back as valid JSON.

---

:::alert{header="Delete the endpoint when you're done" type="warning"}
An endpoint bills per instance-hour for as long as it exists, whether or not you send traffic. An idle `ml.g5.xlarge` can cost more per day than the entire fine-tuning job.

Run the cleanup cells at the bottom of the notebook, then visit the [Clean Up](/04-cleanup/) page to confirm everything is removed.
:::
