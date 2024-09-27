
# car_profile_attributes = {
#     'Front_Height': [770, 715, 880, 935],
#     'Hood_Front_Width': [1160, 1080, 1100, 1388],
#     'Hood_Back_Width': [1460, 1440, 1460, 1520],
#     'Hood_Length': [1070, 1140, 870, 1105],
#     'Hood_Angle': [11, 10, 12.3, 10],
#     'Windscreen_Length': [816, 801, 930, 900],
#     'Windscreen_Angle': [30, 27, 30, 31]
# }

car_profile_attributes = {
    'Bumber_Height':     [770, 500, 600, 580],
    'Front_Hood_Height': [740, 680, 880, 900],
    'Bumber_Hood_Angle': [16, 20, 30, 11],
    'Hood_Length':       [1150, 1200, 760, 1160],
    'Back_Hood_Height':  [980, 920, 1070, 1130],
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