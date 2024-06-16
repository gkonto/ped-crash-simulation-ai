
car_profile_attributes = {
    'Front_Height': [770, 715, 880, 935],
    'Hood_Front_Width': [1160, 1080, 1100, 1388],
    'Hood_Back_Width': [1460, 1440, 1460, 1520],
    'Hood_Length': [1070, 1140, 870, 1105],
    'Hood_Angle': [11, 10, 12.3, 10],
    'Windscreen_Length': [816, 801, 930, 900],
    'Windscreen_Angle': [30, 27, 30, 31]
}

profile2index = {
    "FCR" : 0,
    "RDS" : 1,
    "MPV" : 2,
    "SUV" : 3
}

def expand_car_profiles(row):
    index = profile2index[row["CarProfile"]] 

    for key, value in car_profile_attributes.items():
        row[key] = value[index]
    return row