FROM python:3.8.12
WORKDIR api_file
COPY . ./api_file/
RUN pip install -r ./api_file/fastapi_requriements.txt
RUN python ./api_file/connection.py
CMD uvicorn api_file.main:app