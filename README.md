# Rasa NLU Simple Server

A really simple HTTP server that parses text using [Rasa NLU](https://github.com/RasaHQ/rasa). Models in the `/models` directory are preloaded and cached in memory for fast text parsing.

## Requirements

Python 3.6+

## Installation

After cloning the repository install all the dependencies with pip:

```shell
$ pip3 install -r requirements.txt
```

You will need to install a [spacy language model](https://spacy.io/models) as they are not included. For example:

```shell
$ python -m spacy download en_core_web_sm
```
## Running

Start the server with uvicorn:

```shell
$ uvicorn server:app
```

## Running with docker

You can start the server with the included Dockerfile and docker-compose files but as there is not default language
model it will be of little use. The best way is to create your own Docker image from the base image:

```
FROM ccoreilly/rasa-nlu-microservice

WORKDIR /app

RUN pip install https://github.com/ccoreilly/spacy-catala/releases/download/0.1.0/ca_fasttext_wiki-0.1.0-py3-none-any.whl && \
    python -m spacy link --force ca_fasttext_wiki ca

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "server:app"]
```

You can [configure the gunicorn server](https://docs.gunicorn.org/en/latest/settings.html) running inside the docker image using environment variables.

## Usage

The server exposes the following three endpoints:

### POST `/model/{modelName}/train`
#### Request body
A JSON object following the Rasa NLU [JSON training format](https://rasa.com/docs/rasa/nlu/training-data-format/#json-format)
#### Response body
```json
{
	"message": "Training started"
}
```
### POST `/model/{modelName}/parse`
#### Request body  
```json
{
	"text": "I want to book a flight to Barcelona"
}
```
#### Response body  
```json
{
	"intent": {
		"name": "flight_booking",
		"confidence": 0.9999189376831055
	},
	"entities": [
		{
            "entity": "LOC",
            "value": "Barcelona",
            "start": 12,
            "confidence": null,
            "end": 18,
            "extractor": "SpacyEntityExtractor",
            "processors": [
                "EntitySynonymMapper"
            ]
        }
	],
	"intent_ranking": [
		{
			"name": "flight_booking",
			"confidence": 0.9999189376831055
		},
		{
			"name": "mood_great",
			"confidence": 0.0000021114367427799152
		},
		{
			"name": "goodbye",
			"confidence": 0.000002077534645650303
		},
	],
	"text": "I want to book a flight to Barcelona"
}
```
### GET `/model/{modelName}/status`
#### Response body
```json
{
	"training_status": "TRAINING" | "READY" | "UNKNOWN",
	"training_time": "15.34"
}
```
With training time being the elapsed training time in seconds.