---
title: "Clean Up"
weight: 5
---

:::alert{header="Important" type="warning"}
Complete these cleanup steps to avoid ongoing charges. Resources must be deleted in reverse order of creation.
:::

## Delete Inference Resources

The last cell of the deployment notebook (`lab-1-supervised-fine-tuning/4-deployment.ipynb`) does all of this in one pass, in the order the dependencies require — **inference component first**, then a 45-second wait before the endpoint, because the endpoint will refuse to delete while a component is still attached:

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

Each delete is wrapped individually, so a resource that was never created — or was already removed — is skipped rather than aborting the rest of the cleanup.

:::alert{header="If you deployed to Bedrock instead" type="warning"}
An imported model bills **$1.95/month per Custom Model Unit for storage**, and that continues until you delete the model — independently of whether you ever invoke it. `4a-deployment-bedrock.ipynb` ends with the delete call; run it.
:::

## Verify Cleanup

You can verify that all resources have been deleted by checking the SageMaker console:

1. Go to **SageMaker AI** > **Inference** > **Endpoints** and confirm the endpoint is gone
2. Go to **SageMaker AI** > **Inference** > **Models** and confirm the model is gone
