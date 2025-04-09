import requests
from common.config import (
    RESET_TOKEN_STAGING,
    RESET_TOKEN_PROD,
)
import logging

def resetCache(staging: bool,prod: bool=False):
    if staging:
        url = "https://pegasus-web-staging-j6q4wceq2a-uc.a.run.app/clearCache"
        payload = {}
        headers = {
        'x-api-key': RESET_TOKEN_STAGING
        }

        response = requests.request("GET", url, headers=headers, data=payload)

        logging.info(response.text)
    if prod:
        url = "https://api.forse.io/clearCache"
        payload = {}
        headers = {
        'x-api-key': RESET_TOKEN_PROD
        }

        response = requests.request("GET", url, headers=headers, data=payload)

        logging.info(response.text)
    