from pydantic import BaseSettings
from pydantic import AnyHttpUrl

class Streamlit_Settings(BaseSettings):
    fastapi: AnyHttpUrl
    send_data: AnyHttpUrl
    get_data: AnyHttpUrl

    class Config:
        env_file= './streamlit.env'


streamlit_settings=Streamlit_Settings()
