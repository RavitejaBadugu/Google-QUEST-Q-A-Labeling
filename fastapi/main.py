from fastapi import FastAPI


app=FastAPI()


@app.get('/')
async def welcome_page():
    return 'Starting the api'