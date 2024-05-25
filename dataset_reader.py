import logging
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import settings
from convertibles import DataframeConvertible
from crash_lexer import Lexer
from crash_parser import Parser
from utilities import GifModel, GifPoint, SimulationModel


def create_folder(d: Path):
    directory = os.path.dirname(d)
    if not os.path.exists(directory):
        os.makedirs(directory)


class DatasetReader(object):
    def __init__(self):
        self.logger = logging.getLogger(settings.LOGGER_NAME)
        
    def setLogger(self, logger):
        self.logger.addHandler(logger)

    def read(self):
        raise NotImplementedError("Subclasses must implement the collect method")


class DatasetReaderCSV(DatasetReader, DataframeConvertible):
    __slots__=("filepath", "dataset")

    def __init__(self, fp: Path):
        super().__init__()
        self.filepath = fp
        self.dataset = None
    
    def read(self):
        self.dataset = pd.read_csv(self.filepath)
        self.dataset.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)

    def convert_to_dataframe(self):
        return self.dataset

class SimulationModelReader(object):
    def __init__(self):
        self._path = None
        self._timeseries_labels = set()

    def setPath(self, path):
        self._path = path
    
    def setTimeSeriesLabel(self, lbl):
        self._timeseries_labels.add(lbl)

    def read(self):
        l = Lexer(self._path)
        p = Parser(l)
        for lbl in self._timeseries_labels:
            p.set_accepted_timeseries_name(lbl)
        entry: SimulationModel = p.parse()
        return entry
        

class DatasetFromFiles(DatasetReader):
    def __init__(self, dir: Path):
        super().__init__()
        self.dir = dir
        self.timeseries_labels = set()
    
    def setOutputPath(self, path: Path, overwrite: bool):
        self.output_path = path
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"The path already exists: {path}")
            
    def setTimeSeriesLabel(self, lbl: str):
        self.logger.info(f"Dataset will include timeseries: {lbl}")
        self.timeseries_labels.add(lbl)
    
    def read(self) -> pd.DataFrame:
        if not self.dir.exists():
            raise Exception(f"Directory does not exist: {self.dir}")
        
        self.logger.info(f"Reading dataset from directory: {self.dir}")
        items = [item for item in self.dir.iterdir() if item.is_file()]
        self.logger.info(f"Total number of simulation cases: {len(items)}")

        entries = list()
        for item in tqdm(items, desc="Parsing simulation files", unit="item"):
            reader = SimulationModelReader()
            reader.setPath(item)
            for lbl in self.timeseries_labels:
                reader.setTimeSeriesLabel(lbl)
            entry: SimulationModel = reader.read()
            entries.append(entry.to_dict())
        
        df = pd.DataFrame(entries)
        create_folder(self.output_path)
        df.to_csv(self.output_path, index=True)
        self.logger.info(f"Finished... Results writtern in {self.output_path}")
        

class GifModelReader(object):
    def __init__(self, path: Path):
        self._path = path
        self._name = path.name
        self.trajectory_groups = [
            "Head_XYZ_Coordinate",
            "Sternum_XYZ_Coordinate",
            "Pelvis_XYZ_Coordinate",
            "Right_Knee_XYZ_Coordinate",
            "Left_Knee_XYZ_Coordinate",
            "Right_Ankle_XYZ_Coordinate",
            "Left_Ankle_XYZ_Coordinate",
            "Right_Shoulder_XYZ_Coordinate",
            "Left_Shoulder_XYZ_Coordinate",
            "Right_Elbow_XYZ_Coordinate",
            "Left_Elbow_XYZ_Coordinate",
            "Right_Wrist_XYZ_Coordinate",
            "Left_Wrist_XYZ_Coordinate",
            "Pelvis_Center_XYZ_Coordinate",
            "Pelvis_Left_XYZ_Coordinate",
            "Pelvis_Right_XYZ_Coordinate",
        ]
        self.groups = ["X", "Y", "Z"]

    # Merging function
    def merge_trajectories(self, model: GifModel, trajectories):
        grouped_data = defaultdict(lambda: {'X': [], 'Y': [], 'Z': []})
        
        # Group trajectories by their base names
        for t in trajectories:
            postfix = len("__Coordinate")
            base_name, axis = t.name[: -(postfix+1)], t.name[-postfix]
        
            grouped_data[base_name][axis] = t.values
        
        # Create merged structure
        merged_trajectories = {}
        for base_name, components in grouped_data.items():
            merged_trajectories[f"{base_name}"] = [[x, y, z] for x, y, z in zip(components['X'], components['Y'], components['Z'])]
    
        for name, values in merged_trajectories.items():
            model.addPoint(GifPoint(name, values))
        #merged_trajectories
    

    def read_gif_model(self, m: SimulationModel):
        g = GifModel()
        self.merge_trajectories(g, m.timeseries)
        g.setName(self._name)
        return g


    def read(self):
        reader = SimulationModelReader()
        reader.setPath(self._path)
        timeseries = [trajectory_g.replace("XYZ", group) for trajectory_g in self.trajectory_groups for group in self.groups]
        for lbl in timeseries:
            reader.setTimeSeriesLabel(lbl)
        entry: SimulationModel = reader.read()
    
        return self.read_gif_model(entry)

