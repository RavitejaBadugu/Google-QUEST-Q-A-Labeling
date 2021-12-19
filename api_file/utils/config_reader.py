import yaml
from pydantic import BaseSettings
from typing import Optional

with open('api_file/utils/config.yml') as f:
    parameters=yaml.safe_load(f)

inference_parameters=parameters['INFERENCE_PARAMETERS']['general']

with open('api_file/utils/models.yaml') as f:
    parameters=yaml.safe_load(f)


class DataBaseSettings(BaseSettings):
    POSTGRES_DATABASE_HOSTNAME: str
    DATABASE: str
    TABLES_FEATURES: str
    TABLES_PREDICTIONS: str
    DATABASE_USER: Optional[str]
    DATABASE_PASSWORD: str
    POSTGRES_PORT: int

    class Config:
        env_file='api_file/utils/.env'

database_settings=DataBaseSettings()