import os
from dotenv import load_dotenv

import os
from os.path import join, dirname

dotenv_path = join(dirname(os.path.dirname(__file__)),'..', ".env")
print("loading env from", dotenv_path)

# Load the .env file
load_dotenv(dotenv_path, override=False)


# Set up configurations
DATABASE_NAME = os.getenv("DATABASE_NAME")
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE_URL_MAIN = os.getenv("DATABASE_URL_MAIN")
DATABASE_NAME_postgres=os.getenv("DATABASE_NAME_postgres")

SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')

RESET_TOKEN_STAGING = os.getenv('RESET_TOKEN_STAGING')
RESET_TOKEN_PROD = os.getenv('RESET_TOKEN_PROD')
SLACK_ALERT_HOOK = os.getenv('SLACK_ALERT_HOOK')