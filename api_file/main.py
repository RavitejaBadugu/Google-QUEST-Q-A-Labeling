from fastapi import FastAPI
from api_file import post_api
from api_file.databasemodels import get_n_features,CREATE_FEATURES
from pydantic import create_model

app=FastAPI()

app.include_router(post_api.router)


response_model=create_model('RESPONSE_MODEL',
                            )

@app.get('/')
async def welcome_page():
    return 'Starting the api'

@app.get('/posts')
async def get_posts(n: int):
    posts=get_n_features(n)
    return posts