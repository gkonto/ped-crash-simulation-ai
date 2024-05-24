import logging

import settings
from dataset_reader import DatasetReader


class App(object):
    __slots__=("config", "dataset_reader", "logger")

    def __init__(self):
        self.dataset_reader = None
        self.logger = logging.getLogger(settings.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
    
    def setLogger(self, logger):
        self.logger.addHandler(logger)

    def setReader(self, reader: DatasetReader):
        self.dataset_reader = reader
        self.logger.info("Reader configured")
    
    def run(self):
        if self.dataset_reader is None:
            raise ValueError("Reader cannot be None! Use App::setReader()")
        
        self.dataset_reader.read()
        

    