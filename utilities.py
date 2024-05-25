import os
from enum import Enum
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from tqdm import tqdm


def to_scrollable_table(df: "Dataframe"):
    # Create a scrollable HTML table
    scrollable_table_html = f"""
    <style>
        .scrollable-table {{
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
        }}
        .scrollable-table table {{
            border-collapse: collapse;
            width: 100%;
        }}
        .scrollable-table th, .scrollable-table td {{
            border: 1px solid lightgray;
            padding: 8px;
            text-align: left;
        }}
        .scrollable-table th {{
            position: -webkit-sticky;
            position: sticky;
            top: 0;
            background: gray;
            color: white;
            z-index: 10;
        }}
    </style>
    <div class="scrollable-table">
    {df.to_html(index=False)}
    </div>
    """

    display(HTML(scrollable_table_html))


class Responses:
    __slots__=("hic15", "hic36", 
               "head_z_acc_abs_max", "head_x_acc_abs_max", "head_y_acc_abs_max",
               "bric_abs_max",
               "chest_resultant_acc_max", "chest_resultant_acc_clip_3ms_max")

    def __init__(self):
        self.hic15 = 0
        self.hic36 = 0
        self.head_x_acc_abs_max = 0
        self.head_y_acc_abs_max = 0
        self.head_z_acc_abs_max = 0
        self.bric_abs_max = 0
        self.chest_resultant_acc_max = 0
        self.chest_resultant_acc_clip_3ms_max = 0

    def to_dict(self):
        d = dict()
        d["HIC15_max"] = self.hic15
        d["HIC36_max"] = self.hic36
        d["Head_Z_Acceleration_abs_max"] = self.head_z_acc_abs_max
        d["Head_X_Acceleration_abs_max"] = self.head_x_acc_abs_max
        d["Head_Y_Acceleration_abs_max"] = self.head_y_acc_abs_max
        d["BrIC_abs_max"] = self.bric_abs_max
        d["Chest_Resultant_Acceleration_max"] = self.chest_resultant_acc_max
        d["Chest_Resultant_Acceleration_CLIP3ms_max"] = self.chest_resultant_acc_clip_3ms_max
        return d
    

class Trajectory:
    __slots__=("name", "values")
    def __init__(self, name: str, values: list):
        self.name = name
        self.values = values


class SimulationModel(object):
    class CarProfile(Enum):
        FCR = 1,
        MPV = 2,
        RDS = 3,
        SUV = 4

    class Position(Enum):
        POS = 1

    __slots__=("path", "type", "velocity", "rotation", "translation", "position", "responses", "timeseries")

    def __init__(self, path: str, type: CarProfile, vel: int, transl, rot, pos, responses, timeseries: list):
        self.path : str = path
        self.type : "CarProfile" = type
        self.velocity : int = vel
        self.translation : int = transl
        self.rotation : int = rot
        self.position : str = pos
        self.responses : "Responses" = responses
        self.timeseries: "[Trajectory]" = timeseries
    
    def to_dict(self):
        d = dict()
        d["Path"] = self.path
        d["CarProfile"] = self.type.name
        d["Velocity"] = self.velocity
        d["Translation"] = self.translation
        d["Rotation"] = self.rotation
        d["Position"] = self.position
        d.update(self.responses.to_dict())
        # for timeseries_entry in self.timeseries:
        #     for id, value in enumerate(timeseries_entry.values):
        #         d.update({f"{timeseries_entry.name}_{id}" : value})
        for timeseries_entry in self.timeseries:
            d.update({timeseries_entry.name : timeseries_entry.values})
        return d


class GifPoint(object):
    def __init__(self, name: str, values):
        self._name = name
        self._values = values
    
    def name(self):
        return self._name
    
    def __len__(self):
        return len(self._values)
    
    def __getitem__(self, index: int):
        return self._values[index]


class GifModel(object):
    __slots__=("_points", "_name")

    def __init__(self):
        self._name = ""
        self._points = []
    
    def name(self):
        return self._name

    def setName(self, name: str):
        self._name = name

    def carType(self):
        if not self._name: return ""
        return self._name.split("_")[0].split("=")[1]

    def points(self) -> [GifPoint]: # type: ignore
        return self._points
    
    def addPoint(self, p: GifPoint):
        self._points.append(p)
    
    def getNumTimesteps(self):
        if (len(self._points)) == 0: return 0
        num_timesteps = len(self._points[0])
        for point in self._points:
            if len(point) != num_timesteps:
                raise ValueError("All points must have the same number of timesteps")

        return num_timesteps
        

class GifExportable(object):
    def __init__(self):
        self._temp_folder = None
        self._num_timesteps = 0
        self.lines_to_draw = [
            ("Right_Ankle", "Right_Knee"),
            ("Left_Ankle", "Left_Knee"),
            ("Right_Knee", "Pelvis_Right"),
            ("Left_Knee", "Pelvis_Left"),
            ("Pelvis_Right", "Pelvis_Center"),
            ("Pelvis_Left", "Pelvis_Center"),
            ("Pelvis_Center", "Pelvis"),
            ("Pelvis", "Sternum"),
            ("Sternum", "Head"),
            ("Right_Wrist", "Right_Elbow"),
            ("Left_Wrist", "Left_Elbow"),
            ("Right_Elbow", "Right_Shoulder"),
            ("Left_Elbow", "Left_Shoulder"),
            ("Right_Shoulder", "Sternum"),
            ("Left_Shoulder", "Sternum")
        ]

    def setTempPath(self, p: Path):
        self._temp_folder = p
    
    def setGifOutputPath(self, p: Path):
        self._gif_output_path = p
    
    def setNumTimesteps(self, num: int):
        self._num_timesteps = num

    def createGif(self):
        frames = sorted([f for f in os.listdir(self._temp_folder) if f.endswith('.png')])
        with imageio.get_writer(self._gif_output_path, mode='I', duration=0.01) as writer:
            for frame in frames:
                image = imageio.imread(os.path.join(self._temp_folder, frame))
                writer.append_data(image)

    def drawpoint(self, ax, x, y, z, name: str):
        ax.scatter(x, y, z, name)
        
    def generateFrames(self, gif_model: GifModel):
        if not os.path.exists(self._temp_folder):
            os.makedirs(self._temp_folder)
        
        num_steps = gif_model.getNumTimesteps()
        for t in tqdm(range(0, num_steps, 5), desc="Parsing steps", unit="t"):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            min_b = [float('inf')] * 3 
            max_b = [float('-inf')] * 3
            point_dict = {}
            for point in gif_model.points():
                x, y, z = point[t]
                point_dict[point.name()] = (x, y, z)
                min_b[0] = min(min_b[0], x)
                min_b[1] = min(min_b[1], y)
                min_b[2] = min(min_b[2], z)
                max_b[0] = max(max_b[0], x)
                max_b[1] = max(max_b[1], y)
                max_b[2] = max(max_b[2], z)
                #self.drawpoint(ax, x, y, z, point.name())
                ax.scatter(x, y, z, label=point.name())
            
            if gif_model.carType() == "FCR":
                ax.scatter(0, 0, 0, label="car")

            for (p1name, p2name) in self.lines_to_draw:
                if p1name in point_dict and p2name in point_dict:
                    x1, y1, z1 = point_dict[p1name]
                    x2, y2, z2 = point_dict[p2name]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], color='k') # You can customize the color and style
                #ax.text(x, y, z, point.name()) # add label next to the point
            
            # Calculate ranges
            x_range = max_b[0] - min_b[0]
            y_range = max_b[1] - min_b[1]
            z_range = max_b[2] - min_b[2]
            max_range = max(x_range, y_range, z_range)

            # Calculate middle points
            mid_x = (max_b[0] + min_b[0]) / 2
            mid_y = (max_b[1] + min_b[1]) / 2
            mid_z = (max_b[2] + min_b[2]) / 2

            # Set limits to ensure equal margins
            ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
            ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
            ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

            plt.title(f'Title: {gif_model.name()}')
            plt.figtext(0.5, 0.01,f'Timestep {t}', ha='center', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(f"{self._temp_folder}/frame_{t:03d}.png")
            plt.close()    

    def export(self, gif_model: GifModel):
        self.generateFrames(gif_model)
        self.createGif()
        # TODO remove frames created from generateFrames.


