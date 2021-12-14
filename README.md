# Google-QUEST-Q-A-Labeling
# postgresql docker command
docker run -p 5432:5432 --name postg -e POSTGRES_PASSWORD=yeah -e POSTGRES_USER=admin -e POSTGRES_DB=db postgres:latest
# tensorflow-serving docker command
docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source='C:/Users/ravi1/google_quest_labelling_project/Google-QUEST-Q-A-Labeling/tf-serving/',target=/models/ -t tensorflow/serving --model_config_file=/models/models.config