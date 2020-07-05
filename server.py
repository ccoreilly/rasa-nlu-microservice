import os
import logging

from threading import Thread

from rasa.utils.io import read_yaml, read_config_file

from starlette.applications import Starlette
from starlette.config import Config
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.routing import Route

from train import AsyncTrainer
from cache import InterpreterCache

interpreter_cache = InterpreterCache()
async_trainer = AsyncTrainer(interpreter_cache)
logger = logging.getLogger("RasaNLUSimpleServer")

def preload_models():
    model_dir = "models"
    if os.path.exists(model_dir):
        for (_, _, files) in os.walk(model_dir):
            for file in files:
                interpreter_cache.load(file)

async def parse(request):
    body = await request.json()

    if 'text' not in body:
        raise HTTPException(422)

    model_name = request.path_params["model_name"]
    text = body["text"]
    interpreter = interpreter_cache.load(model_name)

    return JSONResponse(interpreter.parse(text))

async def train(request):
    body = await request.json()

    if 'rasa_nlu_data' not in body:
        raise HTTPException(422)

    model_name = request.path_params["model_name"]
    nlu_data = body["rasa_nlu_data"]

    if 'config' in body:
        config = read_yaml(body['config'])
    else:
        config = read_config_file('./config.yml')
    
    async_trainer.train(nlu_data, model_name, config)

    return JSONResponse({"message": "Training started"})

async def status(request):
    model_name = request.path_params["model_name"]

    status = async_trainer.status(model_name)

    return JSONResponse(status)

preload_models()

routes = [
    Route("/model/{model_name}/parse", endpoint=parse, methods=["POST"]),
    Route("/model/{model_name}/train", endpoint=train, methods=["POST"]),
    Route("/model/{model_name}/status", endpoint=status, methods=["GET"])
]

config = Config();
DEBUG = config('DEBUG', cast=bool, default=False)

app = Starlette(debug=DEBUG, routes=routes)