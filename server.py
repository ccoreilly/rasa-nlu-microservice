import asyncio
import os
import tempfile
import functools
import logging

from threading import Thread

from rasa.nlu.model import Interpreter, components
from rasa.nlu.training_data.formats import MarkdownReader
from rasa.nlu.config import RasaNLUModelConfig
from rasa.model import unpack_model, get_model_subdirectories, create_package_rasa
from rasa.utils.io import read_yaml, read_config_file
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.concurrency import run_in_threadpool

from train import async_train

component_builder = components.ComponentBuilder()
logger = logging.getLogger("RasaNLUSimpleServer")

@functools.lru_cache(maxsize=128)
def load_model(model_path):
    tempdir = tempfile.mkdtemp()
    unpacked_model = unpack_model(model_path, tempdir)
    _, nlu_model = get_model_subdirectories(unpacked_model)
    return Interpreter.load(nlu_model, component_builder)

def preload_models():
    model_dir = "models"
    if os.path.exists(model_dir):
        for (_, _, files) in os.walk(model_dir):
            for file in files:
                load_model(os.path.join("models", file))

async def parse(request):
    try:
        body = await request.json()
        model_name = body["model"]
        text = body["text"]
        model = load_model(os.path.join("models", model_name))
        parsed_text = JSONResponse(model.parse(text))
    except Exception as e:
        logger.error(e)
        parsed_text = JSONResponse(
            {"Status": "500", "Message": "Something went wrong"}
        )
    return parsed_text

async def train(request):
    # try:
    body = await request.json()

    if 'nlu_data' not in body or 'model_name' not in body:
        raise HTTPException(422)

    model_name = body["model_name"]
    nlu_data = body["nlu_data"]

    if 'config' in body:
        config = read_yaml(body['config'])
    else:
        config = read_config_file('./config.yml')
    
    Thread(target=async_train, args=(nlu_data, model_name, config, component_builder)).start()

    return JSONResponse(
            {"Status": "200", "Message": "Training started"}
        )

    # except HTTPException as e:
    #     raise e
    # except Exception as e:
    #     logger.error(e)
    #     parsed_text = JSONResponse(
    #         {"Status": "500", "Message": "Something went wrong"}
    #     )
    # return parsed_text

# preload_models()

routes = [
    Route("/parse", endpoint=parse, methods=["POST"]),
    Route("/train", endpoint=train, methods=["POST"])
]

app = Starlette(debug=True, routes=routes)