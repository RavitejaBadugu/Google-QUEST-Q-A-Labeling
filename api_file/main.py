from fastapi import FastAPI
from api_file import post_api
from pydantic import BaseModel
from api_file.databasemodels import get_n_features
app=FastAPI()

app.include_router(post_api.router)

@app.get('/')
async def welcome_page():
    return 'Starting the api'

class N_data(BaseModel):
    n: int

@app.post('/posts')
async def get_posts(data_to_get: N_data):
    posts=get_n_features(data_to_get.n)
    print(f"posts are {posts}")
    return posts