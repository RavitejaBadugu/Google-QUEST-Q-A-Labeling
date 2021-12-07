from fastapi import FastAPI
from api_file import post_api

app=FastAPI()

app.include_router(post_api.router)

@app.get('/')
async def welcome_page():
    return 'Starting the api'