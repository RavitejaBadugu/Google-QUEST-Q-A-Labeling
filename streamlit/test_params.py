from pydantic import BaseSettings
from pydantic import AnyHttpUrl

class Streamlit_Settings(BaseSettings):
    fastapi_url: AnyHttpUrl
    send_data_url: AnyHttpUrl
    get_data_url: AnyHttpUrl

    class Config:
        env_file= './streamlit.env'


streamlit_settings=Streamlit_Settings()
