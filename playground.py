
from src.data_window import DataWindow


def single_step_prediction():
    single_step_window = DataWindow(input_width=1, label_width=1, shift=1, label_columns=["traffic_volume"])
    # For plotting purposes
    
    pass

if __name__=="__main__":
    single_step_prediction()