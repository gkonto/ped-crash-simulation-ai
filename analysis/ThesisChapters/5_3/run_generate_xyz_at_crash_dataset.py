
import ast
import math
import sys
from pathlib import Path

sys.path.insert(0, "C:\\Users\geork\projects\AIThesis\src\src")
import matplotlib.pyplot as plt

from dataset_reader import DatasetReaderCSV


def find_max_index(acc_list, max_value):
    # print(acc_list)
    # print(type(acc_list))
    indices = [i for i, val in enumerate(acc_list) if abs(float(val)) == max_value]
    if len(indices) == 0:
        return 'Not found'
    elif len(indices) > 1:
        return f'Found multiple: {indices}'
    else:
        return indices[0]
    

def find_accel_abs_index(x_accel, y_accel, z_accel):
    accels_abs = [math.sqrt(x*x + y*y + z*z) for x, y, z in zip(x_accel, y_accel, z_accel)]
    max_value = max(accels_abs)
    max_index = [i for i, x in enumerate(accels_abs) if x == max_value][0]
    return max_value, max_index

def process_row(row):
    row['Head_X_Acceleration_max_index'] = find_max_index(row['Head_X_Acceleration'], float(row['Head_X_Acceleration_abs_max']))
    row['Head_Y_Acceleration_max_index'] = find_max_index(row['Head_Y_Acceleration'], float(row['Head_Y_Acceleration_abs_max']))
    row['Head_Z_Acceleration_max_index'] = find_max_index(row['Head_Z_Acceleration'], float(row['Head_Z_Acceleration_abs_max']))
     
    #accel_max, accel_index = find_accel_abs_index(row["Head_X_Acceleration"], row["Head_Y_Acceleration"], row["Head_Z_Acceleration"])
    # row["MaxAccel"] = accel_max
    # row["MaxAccelIndex"] = accel_index
    return row

def str_to_list(s):
    return ast.literal_eval(s)  

if __name__=="__main__":
    path = Path("C:\\Users\geork\projects\AIThesis\src\\analysis\\3\crash_xyz_coordinate_accel_timeseries.csv")
    reader = DatasetReaderCSV(path)
    reader.read()
    df = reader.convert_to_dataframe()
    df = df.drop(columns=["Position"], errors="ignore")

    df["Head_X_Coordinate"] = df["Head_X_Coordinate"].apply(str_to_list)
    df["Head_Y_Coordinate"] = df["Head_Y_Coordinate"].apply(str_to_list)
    df["Head_Z_Coordinate"] = df["Head_Z_Coordinate"].apply(str_to_list)
    df["Head_X_Acceleration"] = df["Head_X_Acceleration"].apply(str_to_list)
    df["Head_Y_Acceleration"] = df["Head_Y_Acceleration"].apply(str_to_list)
    df["Head_Z_Acceleration"] = df["Head_Z_Acceleration"].apply(str_to_list)

    # Plotting the acceleration of one entry
    # Path entry to find
    path_entry_to_find = "C:\\Users\\geork\\projects\\AIThesis\\datasets\\20240510\\mlres\\Type=MPV_Vel=80_tra=0_rot=0_pos=initial.txt"

    # Find the entry with the specified path
    entry_index = df[df['Path'] == path_entry_to_find].index[0]
    entry = df.iloc[entry_index]

    time_points = list(range(len(entry['Head_X_Acceleration'])))
    plt.figure(figsize=(12, 6))

    plt.plot(time_points, entry['Head_X_Acceleration'], label='X Acceleration')
    plt.plot(time_points, entry['Head_Y_Acceleration'], label='Y Acceleration')
    plt.plot(time_points, entry['Head_Z_Acceleration'], label='Z Acceleration')

    plt.xlabel('Time Points')
    plt.ylabel('Acceleration')
    plt.title(entry["Path"])
    plt.legend()
    plt.grid(True)
    plt.show()

    """
    to_remove_features = ["Id",
                      "HIC36_max", "HIC15_max", 
                      "BrIC_abs_max", 
                      "Chest_Resultant_Acceleration_max", "Chest_Resultant_Acceleration_CLIP3ms_max"]
        # Remove the unwanted columns
    df = df.drop(columns=to_remove_features, errors="ignore")
 
    df["Head_X_Coordinate"] = df["Head_X_Coordinate"].apply(str_to_list)
    df["Head_Y_Coordinate"] = df["Head_Y_Coordinate"].apply(str_to_list)
    df["Head_Z_Coordinate"] = df["Head_Z_Coordinate"].apply(str_to_list)
    df["Head_X_Acceleration"] = df["Head_X_Acceleration"].apply(str_to_list)
    df["Head_Y_Acceleration"] = df["Head_Y_Acceleration"].apply(str_to_list)
    df["Head_Z_Acceleration"] = df["Head_Z_Acceleration"].apply(str_to_list)
    # Apply the function to each row
    df = df.apply(process_row, axis=1)
    #print(df.dtypes)
    
    # Find the maximum and minimum acceleration in the Z axis and their indices
    df['Max_Z_Acceleration'] = df['Head_Z_Acceleration'].apply(lambda x: max(x))
    df['Min_Z_Acceleration'] = df['Head_Z_Acceleration'].apply(lambda x: min(x))
    df['Max_Z_Acceleration_Index'] = df.apply(lambda row: row['Head_Z_Acceleration'].index(row['Max_Z_Acceleration']), axis=1)
    df['Min_Z_Acceleration_Index'] = df.apply(lambda row: row['Head_Z_Acceleration'].index(row['Min_Z_Acceleration']), axis=1)

    # Display the resulting DataFrame
    df = df.drop(columns=["Head_X_Coordinate", "Head_Y_Coordinate", "Head_Z_Coordinate"])
    df = df.drop(columns=["Head_Z_Acceleration", "Head_X_Acceleration", "Head_Y_Acceleration"])
    
    # One-hot encoding the 'CarProfile' column
    try:
        df['CarProfile_orig'] = df['CarProfile']

        df = pd.get_dummies(df, columns=['CarProfile'])
    except:
        pass

    # Create the DataFrame
    data = {
        'Front_Height': [770, 715, 880, 935],
        'Hood_Front_Width': [1160, 1080, 1100, 1388],
        'Hood_Back_Width': [1460, 1440, 1460, 1520],
        'Hood_Length': [1070, 1140, 870, 1105],
        'Hood_Angle': [11, 10, 12.3, 10],
        'Windscreen_Length': [816, 801, 930, 900],
        'Windscreen_Angle': [30, 27, 30, 31]
    }

    # New DataFrame with additional columns based on 'CarProfile'
    index_labels = ['FCR', 'RDS', 'MPV', 'SUV']
    attributes_df = pd.DataFrame(data, index=index_labels)
    try:
        # Map the new columns to original_df based on 'CarProfile'
        for col in attributes_df.columns:
            df[col] = df['CarProfile_orig'].map(attributes_df[col])
        df = df.drop(columns=["CarProfile_orig"])
    except:
        pass
    df.to_csv("C:\\Users\\geork\\projects\\AIThesis\src\\analysis\\3\crash_xyz_coordinate_accel_regres_min_max.csv", index=False)

    #df = df.drop(columns=["Path"], errors="ignore")
    train_df, remaining_df = train_test_split(df, test_size=0.2, random_state=7)
    test_df, val_df = train_test_split(remaining_df, test_size=0.25, random_state=7)
    val_df_exp = expand_dataframe(val_df, ["Head_X_Coordinate", "Head_Y_Coordinate", "Head_Z_Coordinate"])
    print(val_df_exp.shape)
    test_df_exp = expand_dataframe(test_df, ["Head_X_Coordinate", "Head_Y_Coordinate", "Head_Z_Coordinate"])
    print(test_df_exp.shape)
    train_df_exp = expand_dataframe(train_df,  ["Head_X_Coordinate", "Head_Y_Coordinate", "Head_Z_Coordinate"])
    print(train_df_exp.shape)

    val_df_exp.to_csv("C:\\Users\geork\projects\AIThesis\src\datasets\head_coordinates_time_series\\validation_set.csv", index=False)
    test_df_exp.to_csv("C:\\Users\geork\projects\AIThesis\src\datasets\head_coordinates_time_series\\test_set.csv", index=False)
    train_df_exp.to_csv("C:\\Users\geork\projects\AIThesis\src\datasets\head_coordinates_time_series\\train_set.csv", index=False)
    """
    