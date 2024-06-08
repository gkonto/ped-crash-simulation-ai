from enum import Enum

from src.crash_lexer import Lexer
from src.utilities import Responses, SimulationModel, Trajectory


class Errors(object):
    def __init__(self):
        self._errors = []
    
    def hasErrors(self):
        return len(self._errors)
    
    def add(self, err_str: str):
        self._errors.append(err_str)
    
    def errors(self):
        return self._errors


class Parser:
    __slots__=("lexer", "timeseries_to_collect", "history_labels", "errors")

    class ResponsesIndex(Enum):
        HIC15_max = 2
        HIC36_max = 3
        Head_Z_Acceleration_abs_max = 4
        Head_X_Acceleration_abs_max = 5
        Head_Y_Acceleration_abs_max = 6
        BrIC_abs_max = 7
        Chest_Resultant_Acceleration_max = 8
        Chest_Resultant_Acceleration_CLIP3ms_max = 9
    
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.timeseries_to_collect = []
        self.history_labels = set()
        self.errors = Errors()
    
    def set_accepted_timeseries_name(self, timeseries_label):
        self.timeseries_to_collect.append(timeseries_label)
    
    def get_value(self, lines: list, index: ResponsesIndex):
        splitted = lines[index.value].strip().split(",")
        if splitted[1] != index.name:
            raise Exception(f"Expecting {index.name}, got: {splitted[1]} for index: {index.value}")
        
        return round(float(splitted[2]), 3)

    def parse_responses(self, lines: list) -> Responses:
        istart = 1
        iend   = 10
        
        if lines[istart].strip() != "RESPONSES":
            raise Exception(f"Expecting RESPONSES at line 2, got {lines[istart].strip()}")
         
        if lines[iend].strip() != "END":
            raise Exception(f"Expecting END at line 2, got {lines[iend].strip()}") 
        
        response = Responses()
        response.hic15 = self.get_value(lines, Parser.ResponsesIndex.HIC15_max)
        response.hic36 = self.get_value(lines, Parser.ResponsesIndex.HIC36_max)
        response.head_x_acc_abs_max = self.get_value(lines, Parser.ResponsesIndex.Head_X_Acceleration_abs_max)
        response.head_y_acc_abs_max = self.get_value(lines, Parser.ResponsesIndex.Head_Y_Acceleration_abs_max)
        response.head_z_acc_abs_max = self.get_value(lines, Parser.ResponsesIndex.Head_Z_Acceleration_abs_max)
        response.bric_abs_max = self.get_value(lines, Parser.ResponsesIndex.BrIC_abs_max)
        response.chest_resultant_acc_max = self.get_value(lines, Parser.ResponsesIndex.Chest_Resultant_Acceleration_max)
        response.chest_resultant_acc_clip_3ms_max = self.get_value(lines, Parser.ResponsesIndex.Chest_Resultant_Acceleration_CLIP3ms_max)
        return response
    
    def parse_features(self, f: str):
        type_str, vel_str, transl_str, rot_str, pos_str = f.stem.split("_")
        car_type = SimulationModel.CarProfile[type_str.split('=')[1]]
        velocity = int(vel_str.split('=')[1])
        translation = int(transl_str.split('=')[1])
        rotation = int(rot_str.split('=')[1])
        position = pos_str.split('=')[1]
        return (car_type, velocity, translation, rotation, position)

    def parse_trajectories(self, lines: list) -> [Trajectory]:
        trajectories = []
        if not lines[11].startswith("HISTORY"):
            raise Exception(f"Expected line starting with \"HISTORY\" at line 12, got: {lines[11]}")
        self.lexer.cur_line = 11
        i = 1
        while not self.is_eof():
            traj = self.parse_trajectory()  
            if traj:          
                trajectories.append(traj)
                if len(traj.values) != 301:
                    self.errors.add(f"{self.lexer.path}: Error while parsing: Missing values from {traj.name}")
            self.eat()
            i += 1

        return trajectories

    def parse(self):
        type, velocity, translation, rot, pos = self.parse_features(self.lexer.path)
        responses = self.parse_responses(self.lexer.lines)
        trajectories = self.parse_trajectories(self.lexer.lines)
        entry = SimulationModel(self.lexer.path, type, velocity, translation, rot, pos, responses, trajectories)
        return entry, self.errors

    def eat(self):
        self.lexer.cur_line += 1
        while True:
            if self.is_eof():
                break
            elif self.cur_line() == "": # eat whitespaces
                self.lexer.cur_line += 1
            else:
                break
                
    def is_eof(self):
        return self.lexer.is_eof()

    def cur_line(self):
        return self.lexer.lines[self.lexer.cur_line]
    
    def parse_trajectory_name(self):
        return self.cur_line().split(" : ")[-1].strip()

    def parse_trajectory_values(self):
        values = []
        while not self.is_eof() and not self.cur_line().startswith("END"):
            time, val = self.cur_line().split(" , ")
            #values.append((round(float(time), 3), round(float(val), 3)))
            values.append(round(float(val), 3))
            self.eat()
        return values
    
    def timeseries_accepted(self, line):
        return line.split(" : ")[-1].strip() in self.timeseries_to_collect

    def parse_trajectory(self):
        if self.cur_line().startswith("HISTORY"):
            self.history_labels.add(self.parse_trajectory_name())

        if self.cur_line().startswith("HISTORY") and self.timeseries_accepted(self.cur_line()):
            name = self.parse_trajectory_name()
            self.eat()
            values = self.parse_trajectory_values()
            traj = Trajectory(name, values)
            return traj
            