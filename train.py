import tempfile
import os
from threading import Thread, RLock

from rasa.nlu.model import Interpreter, Trainer
from rasa.nlu.train import train as nlu_train
from rasa.nlu.training_data.formats import MarkdownReader
from rasa.nlu.config import RasaNLUModelConfig
from rasa.model import create_package_rasa

from cache import InterpreterCache

class AsyncTrainer():
    def __init__(self, interpreter_cache=InterpreterCache):
        self.interpreter_cache = interpreter_cache
        self.markdown_reader = MarkdownReader()
        self.lock = RLock()
        self.training_status = {}

    def train(self, nlu_data, model_name, config):
        data = self.markdown_reader.reads(nlu_data)
        trainer = Trainer(RasaNLUModelConfig(config), self.interpreter_cache.component_builder)

        thread = Thread(target=self._async_train, args=(trainer, data, model_name))
        thread.start()

    def status(self, model_name):
        with self.lock:
            return self.training_status.get(model_name, "UNKNOWN")

    def _async_train(self, trainer, data, model_name):
        with self.lock:
            self.training_status[model_name] = "TRAINING"
        interpreter = trainer.train(data)
        tempdir = tempfile.mkdtemp()
        trainer.persist(tempdir, None, "nlu")
        
        _model_package = create_package_rasa(tempdir, os.path.join("models", model_name))

        self.interpreter_cache.store(model_name, interpreter)
        
        with self.lock:
            self.training_status[model_name] = "READY"