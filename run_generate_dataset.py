
from pathlib import Path

from src.dataset_reader import DatasetFromFiles

if __name__=="__main__":
    path = Path("C:\\Users\geork\projects\AIThesis\datasets\\20240510\mlres")
    out_path = Path("C:\\Users\geork\projects\AIThesis\src\\datasets\\crash_xyz_coordinate_timeseries.csv")
    r = DatasetFromFiles(path)
    r.setTimeSeriesLabel("Head_X_Coordinate")
    r.setTimeSeriesLabel("Head_Y_Coordinate")
    r.setTimeSeriesLabel("Head_Z_Coordinate")
    r.setOutputPath(out_path, True)
    r.read()