# Rasa NLU Simple Server

A really simple HTTP server that parses text using [Rasa NLU](https://github.com/RasaHQ/rasa). Models in the `/models` directory are preloaded and cached in memory fast text parsing.

## Requirements

Python 3.6+

## Installation

After cloning the repository install all the dependencies with pip:

```shell
$ pip3 install -r requirements.txt
```

## Running

Start the server with uvicorn:

```shell
$ uvicorn server:app
```

The server will respond to POST requests on the `/parse` endpoint with the following body:

```json
{
	"model": "model-filename.tar.gz",
	"text": "I want to book a flight to Barcelona"
}

```

Responses follow the format given by Rasa NLU.