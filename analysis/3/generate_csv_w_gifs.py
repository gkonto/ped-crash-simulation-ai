import ast
import os
import sys
import traceback
from pathlib import Path
from pprint import pprint

import numpy as np

sys.path.insert(0, "C:\\Users\geork\projects\AIThesis\src\src")
import matplotlib.pyplot as plt

from car_profile_expand import expand_car_profiles
from dataset_reader import DatasetReaderCSV

settings = {
    # All
    #"plot_path" : "C:\\Users\\geork\\projects\\AIThesis\\src\\analysis\\3\\plots\\all",
    #"csv_path"  : "C:\\Users\\geork\\projects\\AIThesis\src\\analysis\\3\\datasets\\my_dataset_all.csv",
    #"only_gifs" : False,

    # Only with gifs
    "plot_path" : "C:\\Users\\geork\\projects\\AIThesis\\src\\analysis\\3\\plots\\only_with_gifs",
    "csv_path"  : "C:\\Users\\geork\\projects\\AIThesis\src\\analysis\\3\\datasets\\my_dataset_only_w_gifs.csv",
    "only_gifs" : True,
}
def try_add_gif_path(row, gifs):
    key = row["Path"]
    key = key.split("\\")[-1][:-4]
    gif_path = gifs.get(key, None)
    row["GifPath"] = gif_path
    return row

def read_gif_paths(gifs):
    location = Path("C:\\Users\geork\projects\AIThesis\src\gifs")
    gif_paths = os.listdir(location)
    for gif_path in gif_paths:
        stripped = gif_path[3:-11]
        gifs[stripped] = os.path.join(location, gif_path)

def keep_only_rows_with_gif(df):
    return df[df["GifPath"].notna()]

def str_to_list(s):
    return ast.literal_eval(s)

def find_stop3(z_coords, end):
    if not z_coords or end < 1 or end >= len(z_coords):
        return None
    max_diff = 0
    max_diff_index = end
    for i in range(end, 0, -1):
        diff = abs(z_coords[i] - z_coords[i - 1])
        #print(diff)
        if diff > 5.5: # threshold
            return i
            max_diff = diff
            max_diff_index = i

    return max_diff_index

def find_stop2(accels, end):
    # Check if the list is empty or end is out of range
    if not accels or end < 1 or end >= len(accels):
        return None

    # Initialize variables to keep track of the maximum difference and its index
    max_diff = 0
    max_diff_index = end

    # Compute differences going backwards from end to the start of the list
    for i in range(end, 0, -1):
        diff = abs(accels[i] - accels[i - 1])
        if diff > max_diff:
            max_diff = diff
            max_diff_index = i

    return max_diff_index

def find_stop(z_coords):
    timesteps = len(z_coords)
    timesteps = [i for i in range(301)]
    z_derivative = np.diff(z_coords) / np.diff(timesteps)
    time_derivative = timesteps[:-1]  # The derivative is one element shorter than the original series
    threshold = np.mean(z_derivative) + 2 * np.std(z_derivative)  # Example threshold, adjust as needed
    collision_indices = np.where(z_derivative < -threshold)[0]  # Detecting negative spikes

    if collision_indices.size > 0:
        collision_start_index = collision_indices[0]
        collision_start_time = time_derivative[collision_start_index]
        print(f'Collision likely started at time: {collision_start_time}')
        return collision_start_index

    else:
        print('No collision detected above the threshold.')
        return 0
    

def save_plots(row):
    # Debugging: Print the row to check the values
    print("Row data:", row)
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    try:
        # Ensure the coordinates are arrays
        x_coords = row['Head_X_Coordinate']
        y_coords = row['Head_Y_Coordinate']
        z_coords = row['Head_Z_Coordinate']
        x_accel = row["Head_X_Acceleration"]
        y_accel = row["Head_Y_Acceleration"]
        z_accel = row["Head_Z_Acceleration"]
        #min_z_value = min(z_coords[:-50])
        #min_z_index = z_coords.index(min_z_value)

        #z_desc_index = find_stop3(z_coords, min_z_index)
        t = np.arange(len(z_coords))
        dz_dt = np.gradient(z_coords,t)
        d2z_dt2 = np.gradient(dz_dt, t)
        max_z_2nd_grad = max(d2z_dt2[:-80])
        #impact_time = d2z_dt2.index(max_z_2nd_grad)
        impact_time = np.where(d2z_dt2 == max_z_2nd_grad)[0][0]
        # Ensure they all have the same length (301 timesteps)
        if len(x_coords) == len(y_coords) == len(z_coords) == 301:
            timesteps = range(301)
            
            # X Coordinate plot
            axs[0, 0].plot(timesteps, x_coords, label='X Coordinate')
            axs[0, 0].axvline(x=impact_time, color='r', linestyle='--', label='Min Z')
            #axs[0, 0].axvline(x=z_desc_index, color='g', linestyle='--', label='Impact timestep')
            axs[0, 0].text(impact_time, axs[0, 0].get_ylim()[0], f'{impact_time}', color='r', fontsize=8, rotation=90, va='bottom')
            #axs[0, 0].text(z_desc_index, axs[0, 0].get_ylim()[0], f'{z_desc_index}', color='g', fontsize=8, rotation=90, va='bottom')
            axs[0, 0].set_title('Head X Coordinate')
            axs[0, 0].legend()

            # Y Coordinate plot
            axs[1, 0].plot(timesteps, y_coords, label='Y Coordinate')
            axs[1, 0].axvline(x=impact_time, color='r', linestyle='--', label='Min Z')
            #axs[1, 0].axvline(x=z_desc_index, color='g', linestyle='--', label='Impact timestep')
            axs[1, 0].text(impact_time, axs[1, 0].get_ylim()[0], f'{impact_time}', color='r', fontsize=10, rotation=90, va='bottom')
            #axs[1, 0].text(z_desc_index, axs[1, 0].get_ylim()[0], f'{z_desc_index}', color='g', fontsize=10, rotation=90, va='bottom')
            axs[1, 0].set_title('Head Y Coordinate')
            axs[1, 0].legend()

            # Z Coordinate plot
            axs[2, 0].plot(timesteps, z_coords, label='Z Coordinate')
            axs[2, 0].axvline(x=impact_time, color='r', linestyle='--', label='Min Z')
            #axs[2, 0].axvline(x=z_desc_index, color='g', linestyle='--', label='Impact timestep')
            axs[2, 0].text(impact_time, axs[2, 0].get_ylim()[0], f'{impact_time}', color='r', fontsize=8, rotation=90, va='bottom')
            #axs[2, 0].text(z_desc_index, axs[2, 0].get_ylim()[0], f'{z_desc_index}', color='g', fontsize=8, rotation=90, va='bottom')
            axs[2, 0].set_title('Head Z Coordinate')
            axs[2, 0].legend()
            
            # Calculate accelerations
            #x_accel = np.diff(x_coords, n=2)
            #y_accel = np.diff(y_coords, n=2)
            #z_accel = np.diff(z_coords, n=2)

            
            # Adjust timesteps for acceleration (299 points)
            accel_timesteps = range(301)
            
            # X Acceleration plot
            axs[0, 1].plot(accel_timesteps, x_accel, label='X Acceleration')
            axs[0, 1].axvline(x=impact_time, color='r', linestyle='--')
            #axs[0, 1].axvline(x=z_desc_index, color='g', linestyle='--')
            axs[0, 1].text(impact_time, axs[0, 1].get_ylim()[0], f'{impact_time}', color='r', fontsize=8, rotation=90, va='bottom')
            #axs[0, 1].text(z_desc_index, axs[0, 1].get_ylim()[0], f'{z_desc_index}', color='g', fontsize=8, rotation=90, va='bottom')
            axs[0, 1].set_title('Head X Acceleration')
            #axs[0, 1].legend()

            # Y Acceleration plot
            axs[1, 1].plot(accel_timesteps, y_accel, label='Y Acceleration')
            axs[1, 1].axvline(x=impact_time, color='r', linestyle='--')
            #axs[1, 1].axvline(x=z_desc_index, color='g', linestyle='--')
            axs[1, 1].text(impact_time, axs[1, 1].get_ylim()[0], f'{impact_time}', color='r', fontsize=8, rotation=90, va='bottom')
            #axs[1, 1].text(z_desc_index, axs[1, 1].get_ylim()[0], f'{z_desc_index}', color='g', fontsize=8, rotation=90, va='bottom')
            axs[1, 1].set_title('Head Y Acceleration')
            #axs[1, 1].legend()

            # Z Acceleration plot
            axs[2, 1].plot(accel_timesteps, z_accel, label='Z Acceleration')
            axs[2, 1].axvline(x=impact_time, color='r', linestyle='--')
            #axs[2, 1].axvline(x=z_desc_index, color='g', linestyle='--')
            axs[2, 1].text(impact_time, axs[2, 1].get_ylim()[0], f'{impact_time}', color='r', fontsize=8, rotation=90, va='bottom')
            #axs[2, 1].text(z_desc_index, axs[2, 1].get_ylim()[0], f'{z_desc_index}', color='g', fontsize=8, rotation=90, va='bottom')
            axs[2, 1].set_title('Head Z Acceleration')
            #axs[2, 1].legend()

            axs[1, 2].plot(t, dz_dt, label='Z coord 1st grad')
            axs[1, 2].axvline(x=impact_time, color='r', linestyle='--')
            #axs[1, 2].axvline(x=z_desc_index, color='g', linestyle='--')
            axs[1, 2].text(impact_time, axs[2, 1].get_ylim()[0], f'{impact_time}', color='r', fontsize=8, rotation=90, va='bottom')
            #axs[1, 2].text(z_desc_index, axs[2, 1].get_ylim()[0], f'{z_desc_index}', color='g', fontsize=8, rotation=90, va='bottom')
            axs[1, 2].set_title('Z coord 1st grad')
            #axs[1, 2].legend()

            axs[2, 2].plot(t, d2z_dt2, label='Z coord 2nd grad')
            axs[2, 2].axvline(x=impact_time, color='r', linestyle='--')
            #axs[2, 2].axvline(x=z_desc_index, color='g', linestyle='--')
            axs[2, 2].text(impact_time, axs[2, 1].get_ylim()[0], f'{impact_time}', color='r', fontsize=8, rotation=90, va='bottom')
            #axs[2, 2].text(z_desc_index, axs[2, 1].get_ylim()[0], f'{z_desc_index}', color='g', fontsize=8, rotation=90, va='bottom')
            axs[2, 2].set_title('Z coord 2nd grad')
            #axs[2, 2].legend()

            # Save the plot
            p = row["Path"].split("\\")[-1][:-4]
            plot_filename = os.path.join(settings["plot_path"], f"{p}_plot.png")
            plt.savefig(plot_filename)
            plt.close()
            print(impact_time)
            row["Head_Collision_XYZ_index"] = impact_time
            row["Head_Collision_X"] = x_coords[impact_time]
            row["Head_Collision_Y"] = y_coords[impact_time]
            row["Head_Collision_Z"] = z_coords[impact_time]
            #sys.exit(0)
        else:
            print(f"Error: Coordinate arrays do not all have 301 elements for row {row['Path']}")
            traceback.print_exc()
            sys.exit(1)
    
    except Exception as e:
        # Debugging: Print the error
        print(f"Error while plotting for row {row['Path']}: {e}")
        print(f"Error: Coordinate arrays do not all have 301 elements for row {row['Path']}")
        traceback.print_exc()
        sys.exit(1)
    return row
    

if __name__=="__main__":
    path = Path("C:\\Users\geork\projects\AIThesis\src\\analysis\\3\crash_xyz_coordinate_accel_timeseries.csv")
    reader = DatasetReaderCSV(path)
    reader.read()
    df = reader.convert_to_dataframe()
    df = df.drop(columns=["Position"], errors="ignore")
    gifs = {}
    read_gif_paths(gifs)
    df = df.apply(lambda row: try_add_gif_path(row, gifs), axis=1)

    if settings['only_gifs']:
        df = keep_only_rows_with_gif(df)

    df["Head_X_Coordinate"] = df["Head_X_Coordinate"].apply(str_to_list)
    df["Head_Y_Coordinate"] = df["Head_Y_Coordinate"].apply(str_to_list)
    df["Head_Z_Coordinate"] = df["Head_Z_Coordinate"].apply(str_to_list)

    df["Head_X_Acceleration"] = df["Head_X_Acceleration"].apply(str_to_list)
    df["Head_Y_Acceleration"] = df["Head_Y_Acceleration"].apply(str_to_list)
    df["Head_Z_Acceleration"] = df["Head_Z_Acceleration"].apply(str_to_list)
    
    df = df.apply(save_plots, axis=1)
    df = df.apply(expand_car_profiles, axis=1)
    df = df.drop(columns=["Head_X_Coordinate", "Head_Y_Coordinate", "Head_Z_Coordinate",
                          "Head_X_Acceleration", "Head_Y_Acceleration", "Head_Z_Acceleration",
                          "Head_X_Acceleration_abs_max", "Head_Y_Acceleration_abs_max", "Head_Z_Acceleration_abs_max",
                          "BrIC_abs_max", "Chest_Resultant_Acceleration_max", "Chest_Resultant_Acceleration_CLIP3ms_max",
                          "HIC15_max", "HIC36_max"])

    df.to_csv(settings["csv_path"], index=False)
