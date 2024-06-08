# Regarding notebooks
## PeCr_c_2
This notebook performs binary classification.
Target value: HIC15 (scalar value)
Threshold: 800

## PeCr_c_8
Classification on range of HIC15 values.
Each range has equal number of HIC values.
Target value will be in range from [Q1-Q8]

## PeCr_c_8_1
Classification on range of HIC15 values.
Each range has equal number of HIC values.
Target value will be one-hot-encoded

## PeCr_c_8_2
Same as PeCr_c_8_1.
The difference:
    I analyzed the CarProfiles to their actual attributes:
        - height
        - front length
        - angles 
        - more...

## PeCr_rnn
I am reading the dataset from a file.
The dataset contains timeseries for the XYZ coordinate of the head of the pedestrian.
I will try to predict the next x coordinate, given the previous series.
        