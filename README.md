# Google-QUEST-Q-A-Labeling
## files to be created
# ->tokenizer_folder -path: in api_file folder
It contains 3 files for tokenization. config.json,tokenizer.json and vocab.txt

# ->.env -path: in utils folder

It contains::

POSTGRES_DATABASE_HOSTNAME=None

DATABASE=None

TABLES_FEATURES=None

TABLES_PREDICTIONS=None

DATABASE_USER=None

DATABASE_PASSWORD=None

POSTGRES_PORT=None

# ->db.env -path in api_folder
It contains::

POSTGRES_PASSWORD=None

POSTGRES_USER=None

POSTGRES_DB=None

# -> models.yaml -path: in api_folder
It contains::

HOST: None

PORT: None

VERSION: version number

MODEL_NAMES:

  - fold0

  - fold1

  - fold2

  - fold3

  - fold4

# -> streamlit.env -path: in streamlit folder
It contains::

fastapi_url='http://fastapi_container_name_here:8000/'

send_data_url='http://fastapi_container_name_here/quality'

get_data_url='http://fastapi_container_name_here/posts'

# for below commands we need to create network and all docker containers should run in same network which allows them to connect eachother.
# to create docker network run below command
docker network create network_name

By default the network which we create will be bridge


# postgresql docker command
docker run -p 5432:5432 --name postgres_container_name -d --network=network_name -e POSTGRES_PASSWORD=passwd -e POSTGRES_USER=user_name -e POSTGRES_DB=database_name postgres:latest

# tensorflow-serving docker command
docker run -p 8500:8500 -p 8501:8501 -d --name tensorflow-serving-container-name --network=network_name --mount type=bind,source=complete_path_to_models_folder,target=/models/ -t tensorflow/serving --model_config_file=/models/models.config

if you see  "NET_LOG: Entering the event loop ..." in logs at the end. It means all models are ready for serving.

otherwise check the path you mentioned. Go to tensorflow website or tensorflow-serving github page and see the format the models need 

to be placed in folders
# fastapi docker command
cd api_file

docker build -t fastapi_image_name .

docker run -d --name fastapi_container_name --network=network_name -p 8000:8000 fastapi_image_name

# streamlit docker command
cd streamlit

docker build -t streamlit_image_name .

docker run -d --name streamlit_container_name --network=network_name -p 8502:8502 streamlit_image_name

# photos of streamlit app
* entered question, asnwer and title of a question from stackoverflow which I asked 

![alt text](https://github.com/RavitejaBadugu/Google-QUEST-Q-A-Labeling/blob/development/images/app%20%C2%B7%20Streamlit%20-%20Google%20Chrome%2019-12-2021%2020_11_06.png)

* the model predictions images
![alt text](https://github.com/RavitejaBadugu/Google-QUEST-Q-A-Labeling/blob/development/images/app%20%C2%B7%20Streamlit%20-%20Google%20Chrome%2019-12-2021%2020_11_35.png)
![alt text](https://github.com/RavitejaBadugu/Google-QUEST-Q-A-Labeling/blob/development/images/app%20%C2%B7%20Streamlit%20-%20Google%20Chrome%2019-12-2021%2020_11_42.png)

* here when pressed to show examples. I am getting output in form of dataframe. Here it only shows question, answer,
* title and date when it is posted

![alt text](https://github.com/RavitejaBadugu/Google-QUEST-Q-A-Labeling/blob/development/images/Screenshot%202021-12-19%20201210.png)




