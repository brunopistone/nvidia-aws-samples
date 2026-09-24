---
title: "Getting started with Workshop Studio"
weight: 11
---

## Use the instructor-provided environment

At an AWS instructor-led event, use the temporary AWS account and SageMaker AI Studio environment supplied by the instructor. Follow the steps below to join the event and access that account.

If you are using your own AWS account instead, follow [Your own AWS account](/01-prerequisites/2-account/).

## Before joining

- Have the event link or access code provided by your instructor ready.
- Sign out of unrelated AWS console sessions or use a separate browser profile so you work in the intended account.
- Review the event terms. Do not upload personal data, private contracts, credentials, or other confidential material into the temporary account.
- Confirm the event's AWS Region. Pre-provisioned resources may not exist in another Region even if the console allows you to switch.
- Know when temporary access ends and which non-sensitive workshop materials the instructor permits you to retain.

## Join Workshop Studio

1. Open the instructor-provided event link, or use the event access code and join instructions supplied at the event.
2. Sign in using the offered Workshop Studio sign-in method. When instructed to use email one-time password, select that option.

![Workshop Studio one-time-password sign-in](/static/images_prereq/01-workshopotp.png)

3. Enter your email address, obtain the one-time passcode, and complete sign-in. Keep the passcode and event credentials private.

![Enter the email address for the one-time passcode](/static/images_prereq/01-workshopotp-2email.png)

4. Review the event terms and choose **Join event**.
5. On the event information page, choose **Open AWS console**.

![Open the event AWS console](/static/images_prereq/01-workshop-account.png)

After the console opens, compare the account and Region with the instructor's instructions. If the event link has expired, the account is unavailable, or the expected resources are missing, ask the instructor for help rather than creating substitute infrastructure in a personal account.

## Next step

Continue to [SageMaker AI Studio](/01-prerequisites/3-sagemaker/) to open the supplied profile and JupyterLab space. Event participants do not need to complete the own-account setup unless directed by the instructor.

:button[Open Studio setup instructions]{href="/01-prerequisites/3-sagemaker/"}
