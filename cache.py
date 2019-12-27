import os
import tempfile
from threading import RLock

from rasa.nlu.model import Interpreter, components
from rasa.model import unpack_model, get_model_subdirectories

PREV, NEXT, KEY, RESULT = 0, 1, 2, 3

class InterpreterCache():
    def __init__(self, maxsize = 100):
        self.maxsize = maxsize
        self.interpreters = {}
        self.component_builder = components.ComponentBuilder()
        self.lock = RLock()
        self.root = []  # root of the circular doubly linked list
        self.root[:] = [self.root, self.root, None, None]   # initialize by pointing to self
    
    def load(self, model_name):
        with self.lock:
            link = self.interpreters.get(model_name)

            if link is not None:
                # Move the link to the front of the circular queue
                link_prev, link_next, _name, interpreter = link
                link_prev[NEXT] = link_next
                link_next[PREV] = link_prev
                last = self.root[PREV]
                last[NEXT] = self.root[PREV] = link
                link[PREV] = last
                link[NEXT] = self.root
                return interpreter
        
        interpreter = self._load_model(model_name)
        with self.lock:
            if (model_name in self.interpreters):
                pass
            elif self.interpreters.__len__() >= self.maxsize:
                # Use the old root to store the new key and result.
                oldroot = self.root
                oldroot[KEY] = model_name
                oldroot[RESULT] = interpreter
                # Empty the oldest link and make it the new root.
                # Keep a reference to the old key and old result to
                # prevent their ref counts from going to zero during the
                # update. That will prevent potentially arbitrary object
                # clean-up code (i.e. __del__) from running while we're
                # still adjusting the links.
                self.root = oldroot[NEXT]
                oldkey = self.root[KEY]
                _oldresult = self.root[RESULT]
                self.root[KEY] = self.root[RESULT] = None
                # Now update the cache dictionary.
                del self.interpreters[oldkey]
                # Save the potentially reentrant cache[key] assignment
                # for last, after the root and links have been put in
                # a consistent state.
                self.interpreters[model_name] = oldroot
            else:
                # Put result in a new link at the front of the queue.
                last = self.root[PREV]
                link = [last, self.root, model_name, interpreter]
                last[NEXT] = self.root[PREV] = self.interpreters[model_name] = link
        return interpreter
        
    def _load_model(self, model_name):
        model_path = os.path.join("models", model_name)
        tempdir = tempfile.mkdtemp()
        unpacked_model = unpack_model(model_path, tempdir)
        _, nlu_model = get_model_subdirectories(unpacked_model)
        with self.lock:
            interpreter = Interpreter.load(nlu_model, self.component_builder)
        
        return interpreter

    def store(self, model_name, interpreter):
        with self.lock:
            link = self.interpreters.get(model_name)
            if link is not None:
                link[RESULT] = interpreter
                self.interpreters[model_name] = link
            else:
                last = self.root[PREV]
                link = [last, self.root, model_name, interpreter]
                last[NEXT] = self.root[PREV] = self.interpreters[model_name] = link 