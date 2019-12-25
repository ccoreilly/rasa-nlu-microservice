import tempfile
import os

from rasa.nlu.model import Interpreter, Trainer
from rasa.nlu.train import train as nlu_train
from rasa.nlu.training_data.formats import MarkdownReader
from rasa.nlu.config import RasaNLUModelConfig
from rasa.model import create_package_rasa

def async_train(nlu_data, model_name, config, component_builder):
    markdownReader = MarkdownReader()

    data = markdownReader.reads(nlu_data)

    trainer = Trainer(RasaNLUModelConfig(config), component_builder)

    interpreter = trainer.train(data)
    
    tempdir = tempfile.mkdtemp()
    trainer.persist(tempdir, None, "nlu")
    
    model_package = create_package_rasa(tempdir, os.path.join("models", model_name+".tar.gz"))