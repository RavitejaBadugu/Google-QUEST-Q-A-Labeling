FROM python:3.8.12
WORKDIR api_file
COPY . ./api_file/
RUN pip install -r ./api_file/fastapi_requriements.txt
EXPOSE 8000
CMD python api_file/connection.py && uvicorn api_file.main:app  --host 0.0.0.0 --port 8000