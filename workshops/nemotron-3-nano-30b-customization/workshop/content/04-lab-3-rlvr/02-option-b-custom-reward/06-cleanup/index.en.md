---
title: "Clean Up"
weight: 6
---

:::alert{header="Important" type="warning"}
Complete these cleanup steps to avoid ongoing charges. Resources must be deleted in reverse order of creation.
:::

## Delete Inference Resources

Run the cleanup cells at the end of the deployment notebook (`lab-3a-custom-reward-function-rlvr/5-deployment.ipynb`) to delete the endpoint resources:

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

You can verify that all resources have been deleted by checking the SageMaker console:

1. Go to **SageMaker AI** > **Inference** > **Endpoints** and confirm the endpoint is gone
2. Go to **SageMaker AI** > **Inference** > **Models** and confirm the model is gone

:::alert{header="Note" type="info"}
The custom Reward Function Evaluator you registered in the AI Registry does not incur ongoing charges, but you can remove it from the **Assets** → **Evaluators** view in SageMaker Studio if you no longer need it.
:::
