import tempfile
import os
from threading import Thread, RLock
from timeit import default_timer as timer

from rasa.nlu.model import Trainer
from rasa.nlu.training_data.formats import RasaReader
from rasa.nlu.config import RasaNLUModelConfig
from rasa.model import create_package_rasa

from cache import InterpreterCache

class AsyncTrainer():
    def __init__(self, interpreter_cache=InterpreterCache):
        self.interpreter_cache = interpreter_cache
        self.data_reader = RasaReader()
        self.lock = RLock()
        self.training_status = {}

    def train(self, nlu_data, model_name, config):
        thread = Thread(target=self._async_train, args=(config, nlu_data, model_name))
        thread.start()

    def status(self, model_name):
        with self.lock:
            return self.training_status.get(model_name, {"status": "UNKNOWN"})

    def _async_train(self, config, nlu_data, model_name):
        training_start = timer()
        with self.lock:
            self.training_status[model_name] = {
                "status": "TRAINING",
            }
        
        data = self.data_reader.read_from_json({'rasa_nlu_data': nlu_data})
        with self.interpreter_cache.lock:
            trainer = Trainer(RasaNLUModelConfig(config), self.interpreter_cache.component_builder)
        
        interpreter = trainer.train(data)
        tempdir = tempfile.mkdtemp()
        trainer.persist(tempdir, None, "nlu")
        
        _model_package = create_package_rasa(tempdir, os.path.join("models", model_name))

        self.interpreter_cache.store(model_name, interpreter)
        
        with self.lock:
            training_end = timer()
            self.training_status[model_name] = {
                "status": "READY",
                "training_time": f"{training_end - training_start:.2f}"
            }