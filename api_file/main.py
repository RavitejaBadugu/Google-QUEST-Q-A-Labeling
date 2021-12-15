from fastapi import FastAPI
from api_file import post_api
from api_file.databasemodels import get_n_features
app=FastAPI()

app.include_router(post_api.router)

@app.get('/')
async def welcome_page():
    return 'Starting the api'

@app.get('/posts')
async def get_posts(n: int):
    posts=get_n_features(n)
    return posts