"""Module for database utilities"""
# Standard
import os
import json
import logging
# Installed
import boto3

logger = logging.getLogger(__name__)


def set_db_credentials_from_secret_manager(secret_name: str):
    """Set Environment Variables for RDS access
    Parameters
    ----------
    secret_name : str
        The name of the secret in the Secrets Manager to access.
    """
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name="us-west-2")
    secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    secret_object = json.loads(secret_value_response['SecretString'])

    os.environ["LIBERA_DB_HOST"] = secret_object["host"]
    os.environ["LIBERA_DB_USER"] = secret_object["username"]
    os.environ["LIBERA_DB_NAME"] = secret_object["dbname"]
    os.environ["PGPASSWORD"] = secret_object["password"]
    logger.debug("Secret loaded and stored as environment variables.")
