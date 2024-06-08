from src.dataset_reader import DatasetReader


class DatasetPreprocessor(object):
    __slots__=("dataset_reader")

    def __init__(self):
        self.dataset_reader = None

    def setReader(self, reader: DatasetReader):
        self.dataset_reader = reader