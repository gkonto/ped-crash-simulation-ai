
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.dataset_reader import DatasetReaderCSV


def calculate_magnitude(accel_x, accel_y, accel_z, row):
    # Convert string representation of lists to actual listsprint
    if type(accel_y) != str:
        print(row["Path"])
    accel_x = np.array(eval(accel_x))
    accel_y = np.array(eval(accel_y))
    accel_z = np.array(eval(accel_z))
    
    # Calculate the magnitude of acceleration at each point
    magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    return magnitude


class CollisionDetector(object):
    def __init__(self):
        self.dataset = None
    
    def reset(self):
        self.dataset = None

    def read_accelerations_from_file(self, path: Path):
        reader = DatasetReaderCSV(path)
        reader.read()
        self.dataset = reader.convert_to_dataframe()
    
    def calculate_acceleration_magnitute(self):
        # Apply the function to each row
        self.dataset['Acceleration_Magnitude'] = self.dataset.apply(
            lambda row: calculate_magnitude(row['Head_X_Acceleration'], row['Head_Y_Acceleration'], row['Head_Z_Acceleration'], row),
            axis=1)
    
    def export_pngs(self):
        # Plot the acceleration magnitude and save as PNG
        for index, row in self.dataset.iterrows():
            plt.figure()
            plt.plot(row['Acceleration_Magnitude'])
            plt.xlabel('Time (ms)')
            plt.ylabel('Acceleration Magnitude (m/s^2)')
            plt.title(f'Acceleration Magnitude for Entry {index}')
            plt.savefig(f'acceleration_plot_{index}.png')
            plt.close()
            break
        

if __name__=="__main__":

    x = CollisionDetector()
    x.read_accelerations_from_file(Path("C:\\Users\geork\projects\AIThesis\src\datasets\crash_accelerations.csv"))
    x.calculate_acceleration_magnitute()
    x.export_pngs()

    # path = Path("C:\\Users\geork\projects\AIThesis\src\datasets\crash_accelerations.csv")
    # reader = DatasetReaderCSV(path)

    # reader.read()
    # df_accelerations = reader.convert_to_dataframe()
    # df_less_than_301 = df_accelerations[
    # df_accelerations.apply(
    #     lambda row: has_less_than_301_entries(
    #         row['Head_X_Acceleration'], 
    #         row['Head_Y_Acceleration'], 
    #         row['Head_Z_Acceleration']
    #     ), 
    #     axis=1
    #     )
    # ]
    # # Optional: Reset index for the new DataFrame
    # df_less_than_301 = df_less_than_301.reset_index(drop=True)

    # # Print the new DataFrame or save it if necessary
    # print(df_less_than_301)
    # ids_to_remove = df_less_than_301['Id'].unique()

    # # Drop rows with these Ids from the original DataFrame
    # df_accelerations_cleaned = df_accelerations[~df_accelerations['Id'].isin(ids_to_remove)]

    # # Optional: Reset index for the cleaned DataFrame
    # df_accelerations_cleaned = df_accelerations_cleaned.reset_index(drop=True)

    # # Print the cleaned DataFrame or save it if necessary
    # print("Cleaned DataFrame:")
    # print(df_accelerations_cleaned)

    # #utilities.to_scrollable_table(df_accelerations.drop(columns=["Position"], errors="ignore"))
    # #print(df_accelerations.head)
    # # Function to calculate the magnitude of acceleration
    # def calculate_magnitude(accel_x, accel_y, accel_z, row):
    #     # Convert string representation of lists to actual listsprint
    #     if type(accel_x) != str:
    #         print(type(accel_x))
    #         print(accel_x)
    #         print(row["Path"])
    #         print(row)

    #     accel_x = np.array(eval(accel_x))
    #     accel_y = np.array(eval(accel_y))
    #     accel_z = np.array(eval(accel_z))
        
    #     # Calculate the magnitude of acceleration at each point
    #     magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    #     return magnitude

    # # Apply the function to each row
    # df_accelerations_cleaned['Acceleration_Magnitude'] = df_accelerations_cleaned.apply(
    #     lambda row: calculate_magnitude(row['Head_X_Acceleration'], row['Head_Y_Acceleration'], row['Head_Z_Acceleration'], row),
    #     axis=1
    # )

    # # Plot the acceleration magnitude and save as PNG
    # for index, row in df_accelerations_cleaned.iterrows():
    #     plt.figure()
    #     plt.plot(row['Acceleration_Magnitude'])
    #     plt.xlabel('Time (ms)')
    #     plt.ylabel('Acceleration Magnitude (m/s^2)')
    #     plt.title(f'Acceleration Magnitude for Entry {index}')
    #     plt.savefig(f'acceleration_plot_{index}.png')
    #     plt.close()
    #     break
    