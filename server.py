import os
import tempfile
import functools
import logging

from rasa.nlu.model import Interpreter, components
from rasa.model import unpack_model, get_model_subdirectories
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse


componentBuilder = components.ComponentBuilder()
logger = logging.getLogger("RasaNLUSimpleServer")

@functools.lru_cache(maxsize=128)
def load_model(model_path):
    tempdir = tempfile.mkdtemp()
    unpacked_model = unpack_model(model_path, tempdir)
    _, nlu_model = get_model_subdirectories(unpacked_model)
    return Interpreter.load(nlu_model, componentBuilder)

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

preload_models()

routes = [Route("/parse", endpoint=parse, methods=["POST"])]
app = Starlette(debug=True, routes=routes)