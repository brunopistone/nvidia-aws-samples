---
title: "Your own AWS account"
weight: 12
---

::alert[At an AWS-led event, use [Workshop Studio access](/01-prerequisites/1-workshop/) instead. Running this workshop in your own account incurs AWS charges for the resources you use.]{type="warning"}

## 1. Choose the account and Region

Use a dedicated account or an isolated environment approved by your administrator. If needed, follow [Setting Up Your AWS Environment](https://aws.amazon.com/getting-started/guides/setup-environment/). Use one Region consistently for Studio, S3, training, and hosting. The workshop supports setup in `us-east-1` and `us-west-2`; confirm that the models and services required for your selected lab are available in the Region you choose.

## 2. Set up the Studio domain and execution role

Use an existing SageMaker AI domain and user profile, or follow [Studio quick setup](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html). Identify the profile's [execution role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html#sagemaker-roles-get-execution-role). This service role is separate from the identity you use to sign in to the console. The notebooks resolve the Studio execution role with `get_execution_role()` and use a role named `sagemaker_execution_role` as a fallback. If your environment uses that fallback, the role must already exist with the required permissions.

The repository includes a [workshop CloudFormation template](/static/cfn/workshop-setup.yaml) that provisions the Studio environment, workshop networking, and one MLflow App using the existing Studio bucket and execution role. The serverless MLflow App is standalone and requires no domain. Have an administrator review its infrastructure and workshop-oriented permissions before deploying it. It is not a least-privilege production policy, and a successful stack does not automatically grant model access, service quotas, or access through organization-level controls.

If using the template, wait for `CREATE_COMPLETE` or `UPDATE_COMPLETE`. Leave `MLFLOW_APP_ARN` empty for automatic discovery of the uniquely matching tagged stack app in either journey. Copy its `MlflowAppArn` output into the selected journey's `config.py` only to override discovery or resolve ambiguity. A CloudFormation-precreated app is a startup optimization, not a notebook prerequisite. CloudFormation manages the app's lifecycle through the native [AWS::SageMaker::MlflowApp resource](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-sagemaker-mlflowapp.html); notebooks check readiness before using it. Automatic registration from MLflow into SageMaker Model Registry is explicitly disabled: serverless training registers its own Model Package, while the Training-job track deploys the S3 artifact.

The template's networking map covers 23 Regions, not 23 fully supported workshop environments. It skips MLflow provisioning in `af-south-1`, `ap-east-1`, `eu-south-1`, `ap-northeast-3`, `ap-southeast-3`, `me-south-1`, and `me-central-1`; `MlflowProvisioning` reports that limitation. Use `us-east-1` or `us-west-2` for this workshop and check model availability separately.

Without the template, either configure an administrator-approved [MLflow App](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateMlflowApp.html) ARN in the notebook Region or leave `MLFLOW_APP_ARN` empty to enable automatic reuse/create. The primary Training jobs notebook reuses or creates `nemotron-contractnli-workshop`; optional serverless notebooks use native SDK discovery/create unless creation overrides select the named workshop path. Both wait for readiness without requiring a domain or legacy Tracking Server. An invalid or inaccessible explicitly configured ARN fails setup without creating a replacement. See [MLflow setup behavior](/01-prerequisites/3-sagemaker/#7-reuse-or-create-the-workshop-mlflow-app) for defaults, overrides, and timeout recovery.

The notebook identity needs `sagemaker:DescribeMlflowApp` and `sagemaker:CreatePresignedMlflowAppUrl`; automatic reuse/create additionally needs `sagemaker:ListMlflowApps`, `sagemaker:ListTags`, `sagemaker:CreateMlflowApp`, and `iam:PassRole` for the app role. The app role must trust SageMaker and access its approved S3 artifact store, including KMS permissions where applicable. Training/evaluation roles need the relevant `sagemaker-mlflow` tracking permissions. Native serverless creation also needs the SDK's S3 bucket/prefix permissions. Have your administrator review these against the chosen setup path rather than granting broad access to work around a failure.

## 3. Configure IAM and service access

Check role trust, identity permissions, and role passing separately. A role's **trust policy** determines who may assume it. Its **identity policy** determines the actions it can perform. The **submitting identity** needs permission to pass the intended service role where the API requires it. Bucket policies, key policies, permission boundaries, and organization controls can impose additional restrictions.

| Actor or resource            | Access to verify                                                                                                                                                                                                                                                                                               |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Notebook/submitting identity | Submit, describe, and stop SageMaker training jobs; create, describe, and delete hosting resources; invoke workshop endpoints; and pass the approved execution role. Include customization, registry dataset and evaluator, model package, and managed pipeline permissions when required by the selected lab. |
| SageMaker execution role     | Trust `sagemaker.amazonaws.com`; read the approved source and input data, write outputs, and load model artifacts.                                                                                                                                                                                             |
| S3                           | Allow bucket listing and object access separately: `ListBucket` applies to the bucket, while read/write permissions apply to approved object prefixes.                                                                                                                                                         |
| Container images             | Permit the configured runtime to obtain its training and serving images.                                                                                                                                                                                                                                       |
| CloudWatch                   | Allow services to write logs and participants to read the relevant logs.                                                                                                                                                                                                                                       |
| Customer-managed KMS keys    | Allow the required cryptographic operations in both the role policy and key policy for encrypted data and artifacts.                                                                                                                                                                                           |

Ask your administrator to scope permissions to the workshop resource names, S3 prefixes, images, and roles. If access is denied, identify the denied action, caller, and resource before changing a policy.

### Model and Amazon Bedrock access

Review the licenses and access terms for the configured NVIDIA Nemotron models before running the labs. Confirm access to the selected model through SageMaker JumpStart or Hugging Face, as applicable, including support for the required customization recipe. Keep any model-download credentials out of notebook output, committed files, and shared configuration.

Amazon Bedrock access is also required for the model invocation and managed evaluation activities that use it. Verify access to the models and inference profiles configured in the selected lab's evaluation notebook in the intended Bedrock Region. SageMaker permissions do not grant Bedrock model access or invocation permissions.

For Bedrock evaluation jobs, the submitting identity needs the relevant evaluation-job permissions and `iam:PassRole` for the approved service role. That role must trust `bedrock.amazonaws.com` and have model invocation permissions and access to the S3 inputs and outputs, including KMS permissions where applicable. If a role is shared with SageMaker, it must have the required trust and permissions for both services. Use the [Bedrock evaluation service role guidance](https://docs.aws.amazon.com/bedrock/latest/userguide/judge-service-roles.html) when reviewing these policies with your administrator.

## 4. Check service quotas

In the selected Region, use **Service Quotas** to check the following resources before launching the labs. Request increases in advance if your account's quotas are below the required values.

| Use                                             | Required resource                                                                                                            | Quota to verify                                             |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Studio JupyterLab                               | One `ml.m5.large` app by default, or the template's selected `JupyterLabInstanceType`                                        | JupyterLab app quota for that instance type                 |
| Serverless customization hosting                | One `ml.g5.12xlarge` endpoint instance                                                                                       | Endpoint instance quota                                     |
| Serverless customization and managed evaluation | Managed customization and evaluation jobs                                                                                    | Applicable service and model limits                         |
| Training jobs                                   | One instance selected from `ml.g5.2xlarge`, `ml.g6.2xlarge`, `ml.g6e.2xlarge`, `ml.g5.xlarge`, `ml.g6.xlarge`, in that order | Training instance quota of at least one for every candidate |
| Optional training warm pool                     | Disabled by default (`KEEP_ALIVE_SECONDS = 0`); if enabling retention, check the intended training candidate types           | Separate training warm-pool quota                           |
| Training-job model hosting                      | One instance across pools prioritized as `ml.g5.xlarge`, `ml.g6.xlarge`, `ml.g6.2xlarge`, `ml.g6e.xlarge`                    | Check endpoint instance quota for each intended pool type   |

For [training Instance Preferences](https://docs.aws.amazon.com/sagemaker/latest/dg/train-instance-preferences.html), SageMaker checks training quota for every candidate, not just the selected type. A quota below the requested count for any candidate rejects the request, so check all five before submission. Hosting uses a different candidate list; verify hosting quota for the intended pools rather than reusing the training quota check.

Training, warm-pool, hosting, and Studio quotas are separate and specific to the account, Region, instance type, and usage category. See the [JupyterLab quota guidance](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-admin-guide-quotas.html) for Studio requirements. An approved quota does not guarantee immediate capacity.

## 5. Check network and storage access

Notebook and training environments need approved access to the source repository, the public ContractNLI download, Python packages, relevant Hugging Face assets, S3, and container images. Private networking must provide the required service endpoints or approved outbound route. Verify access from both Studio and the remote training environment, since their network configurations can differ.

Identify the S3 bucket and prefix approved for the workshop, and confirm that the execution role can read inputs and write outputs there. The notebook setup cells display the selected bucket and Region. Check those values before uploading data, and use only workshop data or other material approved for the account.

## Ready to open Studio

Confirm the account and Region, Studio profile and execution role, model and service access, quotas, and network and storage permissions with your administrator. Then continue to [SageMaker AI Studio](/01-prerequisites/3-sagemaker/).
