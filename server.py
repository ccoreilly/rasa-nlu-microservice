import os
import tempfile
import functools
import logging

from rasa.nlu.model import Interpreter, components
from rasa.model import unpack_model, get_model_subdirectories
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse


class RasaNLUSimpleServer:
    def __init__(self):
        self.componentBuilder = components.ComponentBuilder()
        self.logger = logging.getLogger("RasaNLUSimpleServer")

        self.preload_models()

        self.routes = [Route("/parse", endpoint=self.parse, methods=["POST"])]
        self.app = Starlette(debug=True, routes=self.routes)

    @functools.lru_cache(maxsize=128)
    def load_model(self, model_path):
        tempdir = tempfile.mkdtemp()
        unpacked_model = unpack_model(model_path, tempdir)
        _, nlu_model = get_model_subdirectories(unpacked_model)
        return Interpreter.load(nlu_model, self.componentBuilder)

    def preload_models(self):
        model_dir = "models"
        if os.path.exists(model_dir):
            for (here, dirs, files) in os.walk(model_dir):
                for file in files:
                    self.load_model(os.path.join("models", file))

    async def parse(self, request):
        try:
            body = await request.json()
            model_name = body["model"]
            text = body["text"]
            model = self.load_model(os.path.join("models", model_name))
            parsed_text = JSONResponse(model.parse(text))
        except Exception as e:
            self.logger.error(e)
            parsed_text = JSONResponse(
                {"Status": "500", "Message": "Something went wrong"}
            )
        return parsed_text


app = RasaNLUSimpleServer().app
