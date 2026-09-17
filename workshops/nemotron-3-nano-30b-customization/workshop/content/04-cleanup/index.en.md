---
title: "Clean Up"
weight: 40
---

:::alert{header="Important" type="warning"}
Complete these cleanup steps to avoid ongoing charges. Inference resources must be deleted in reverse order of creation.
:::

## What actually costs money

| Resource                                        | Charged while it exists?                                                            |
| ----------------------------------------------- | ----------------------------------------------------------------------------------- |
| SageMaker real-time endpoint (`ml.g5.xlarge`) | **Yes** — per instance-hour, whether or not you send requests                       |
| Serverless training / evaluation jobs           | No — they run to completion and stop                                                |
| SageMaker AI Datasets, Model Package Groups     | Only the underlying S3 storage                                                      |
| JupyterLab space                                | Per instance-hour while the space is running                                        |

The endpoint is the expensive one. Delete it first.

## Delete SageMaker Inference Resources

Run the final cells of **`code/4-deployment.ipynb`** to delete the endpoint resources, in this order:

### 1. Delete Inference Component

```python
from sagemaker.core.resources import InferenceComponent

InferenceComponent.get(inference_component_name=ic_name).delete()
```

### 2. Delete Model

```python
from sagemaker.core.resources import Model

Model.get(model_name=model_name).delete()
```

### 3. Delete Endpoint

```python
from sagemaker.core.resources import Endpoint

Endpoint.get(endpoint_name=endpoint_name).delete()
```

### 4. Delete Endpoint Configuration

```python
from sagemaker.core.resources import EndpointConfig

EndpointConfig.get(endpoint_config_name=endpoint_config_name).delete()
```

## Verify Cleanup

Confirm in the SageMaker AI console that nothing is left running:

1. **SageMaker AI** → **Inference** → **Endpoints** — the endpoint is gone
2. **SageMaker AI** → **Inference** → **Models** — the model is gone
3. **SageMaker AI** → **Inference** → **Endpoint configurations** — the config is gone

## Optional: remove registry entries and artifacts

These do not incur compute charges, but you may want to remove them:

- **Assets** → **Datasets** — the three `Multilingual-Thinking-sft-*` datasets
- **Models** — the `*-sft-mpg` Model Package Group and its versions
- The S3 default bucket prefixes holding the datasets, training output and evaluation results

:::alert{header="Shut down your JupyterLab space" type="info"}
If you are running in your own account, stop the JupyterLab space when you are done — it bills per instance-hour while running.
:::
