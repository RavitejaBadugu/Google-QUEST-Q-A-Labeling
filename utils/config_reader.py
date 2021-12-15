import yaml
from pydantic import BaseSettings
from typing import Optional

with open('utils/config.yml') as f:
    parameters=yaml.safe_load(f)

inference_parameters=parameters['INFERENCE_PARAMETERS']['general']

with open('utils/models.yaml') as f:
    parameters=yaml.safe_load(f)


class DataBaseSettings(BaseSettings):
    HOSTNAME: str
    DATABASE: str
    TABLES_FEATURES: str
    TABLES_PREDICTIONS: str
    DATABASE_USER: Optional[str]
    DATABASE_PASSWORD: str
    PORT: int

    class Config:
        env_file='utils/.env'

database_settings=DataBaseSettings()