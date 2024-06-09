
import ast
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset_reader import DatasetReaderCSV


def expand_dataframe(df, timeseries_cols):
    expanded_data = []

    for _, row in df.iterrows(): # for each row
        timeseries_lists = [row[col] for col in timeseries_cols]
        max_len = max(len(ts) for ts in timeseries_lists) # 301
        for i in range(max_len): # 301
            expanded_row = row.drop(timeseries_cols).to_dict()
            for col, timeseries in zip(timeseries_cols, timeseries_lists):
                expanded_row[col] = timeseries[i] if i < len(timeseries) else None
            expanded_row["timestep"] = i
            expanded_data.append(expanded_row)
    return pd.DataFrame(expanded_data)


if __name__=="__main__":
    path = Path("C:\\Users\geork\projects\AIThesis\src\datasets\crash_xyz_coordinate_timeseries.csv")
    reader = DatasetReaderCSV(path)
    reader.read()
    df = reader.convert_to_dataframe()
    df = df.drop(columns=["Position"], errors="ignore")
    
    to_remove_features = ["Id",
                      "HIC36_max", "HIC15_max", 
                      "Head_Z_Acceleration_abs_max", "Head_X_Acceleration_abs_max", "Head_Y_Acceleration_abs_max",
                      "BrIC_abs_max", 
                      "Chest_Resultant_Acceleration_max", "Chest_Resultant_Acceleration_CLIP3ms_max"]
    # Remove the unwanted columns
    df = df.drop(columns=to_remove_features, errors="ignore")
    def str_to_list(s):
        return ast.literal_eval(s)

    df["Head_X_Coordinate"] = df["Head_X_Coordinate"].apply(str_to_list)
    df["Head_Y_Coordinate"] = df["Head_Y_Coordinate"].apply(str_to_list)
    df["Head_Z_Coordinate"] = df["Head_Z_Coordinate"].apply(str_to_list)

    # One-hot encoding the 'CarProfile' column
    try:
        df['CarProfile_orig'] = df['CarProfile']

        df = pd.get_dummies(df, columns=['CarProfile'])

        # Display the DataFrame after one-hot encoding
        # print("\nDataFrame after One-hot Encoding:")
        # print(df)
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

        # Drop the 'CarProfile' column
        # df = df.drop(columns=['CarProfile'])
        df = df.drop(columns=["CarProfile_orig"])
    except:
        pass

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