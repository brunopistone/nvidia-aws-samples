import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

IAM_USER_PREFIX = "BedrockAPIKey-"


def lambda_handler(event, context):
    iam = boto3.client("iam")
    deleted_users = []
    errors = []

    paginator = iam.get_paginator("list_users")
    for page in paginator.paginate():
        for user in page["Users"]:
            username = user["UserName"]
            if not username.startswith(IAM_USER_PREFIX):
                continue

            logger.info(f"Found target user: {username}")
            try:
                _delete_access_keys(iam, username)
                _delete_login_profile(iam, username)
                _detach_policies(iam, username)
                _delete_inline_policies(iam, username)
                _remove_from_groups(iam, username)
                _delete_mfa_devices(iam, username)
                _delete_signing_certificates(iam, username)
                _delete_ssh_public_keys(iam, username)
                _delete_service_specific_credentials(iam, username)
                iam.delete_user(UserName=username)
                logger.info(f"Deleted user: {username}")
                deleted_users.append(username)
            except Exception as e:
                msg = f"Failed to delete {username}: {e}"
                logger.error(msg)
                errors.append(msg)

    summary = {
        "deleted_users": deleted_users,
        "deleted_count": len(deleted_users),
        "errors": errors,
    }
    logger.info(f"Cleanup complete: {summary}")
    return summary


def _delete_access_keys(iam, username):
    for key in iam.list_access_keys(UserName=username)["AccessKeyMetadata"]:
        iam.delete_access_key(UserName=username, AccessKeyId=key["AccessKeyId"])
        logger.info(f"  Deleted access key {key['AccessKeyId']}")


def _delete_login_profile(iam, username):
    try:
        iam.delete_login_profile(UserName=username)
        logger.info(f"  Deleted login profile for {username}")
    except iam.exceptions.NoSuchEntityException:
        pass


def _detach_policies(iam, username):
    for p in iam.list_attached_user_policies(UserName=username)["AttachedPolicies"]:
        iam.detach_user_policy(UserName=username, PolicyArn=p["PolicyArn"])
        logger.info(f"  Detached policy {p['PolicyName']}")


def _delete_inline_policies(iam, username):
    for name in iam.list_user_policies(UserName=username)["PolicyNames"]:
        iam.delete_user_policy(UserName=username, PolicyName=name)
        logger.info(f"  Deleted inline policy {name}")


def _remove_from_groups(iam, username):
    for g in iam.list_groups_for_user(UserName=username)["Groups"]:
        iam.remove_user_from_group(UserName=username, GroupName=g["GroupName"])
        logger.info(f"  Removed from group {g['GroupName']}")


def _delete_mfa_devices(iam, username):
    for device in iam.list_mfa_devices(UserName=username)["MFADevices"]:
        serial = device["SerialNumber"]
        iam.deactivate_mfa_device(UserName=username, SerialNumber=serial)
        if serial.startswith("arn:"):
            iam.delete_virtual_mfa_device(SerialNumber=serial)
        logger.info(f"  Deleted MFA device {serial}")


def _delete_signing_certificates(iam, username):
    for cert in iam.list_signing_certificates(UserName=username)["Certificates"]:
        iam.delete_signing_certificate(UserName=username, CertificateId=cert["CertificateId"])
        logger.info(f"  Deleted signing certificate {cert['CertificateId']}")


def _delete_ssh_public_keys(iam, username):
    for key in iam.list_ssh_public_keys(UserName=username)["SSHPublicKeys"]:
        iam.delete_ssh_public_key(UserName=username, SSHPublicKeyId=key["SSHPublicKeyId"])
        logger.info(f"  Deleted SSH public key {key['SSHPublicKeyId']}")


def _delete_service_specific_credentials(iam, username):
    for cred in iam.list_service_specific_credentials(UserName=username)["ServiceSpecificCredentials"]:
        iam.delete_service_specific_credential(
            UserName=username,
            ServiceSpecificCredentialId=cred["ServiceSpecificCredentialId"]
        )
        logger.info(f"  Deleted service-specific credential {cred['ServiceSpecificCredentialId']}")
