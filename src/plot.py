import random

import matplotlib.pyplot as plt
import pandas as pd


def plot_timeseries_random_entry(df, label1, label2, label3):
    # Select a random index
    random_index = random.randint(0, len(df) - 1)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(4, 8), sharex=True)

    # Plotting Head_X_Coordinate
    axes[0].plot(range(len(df.loc[random_index, label1])), df.loc[random_index, label1], label=f'Rotation {df.loc[random_index, "Rotation"]}')
    axes[0].set_title(label1)
    axes[0].set_ylabel(label1)

    # Plotting Head_Y_Coordinate
    axes[1].plot(range(len(df.loc[random_index, label2])), df.loc[random_index, label2], label=f'Rotation {df.loc[random_index, "Rotation"]}')
    axes[1].set_title(label2)
    axes[1].set_ylabel(label2)

    # Plotting Head_Z_Coordinate
    axes[2].plot(range(len(df.loc[random_index, label3])), df.loc[random_index, label3], label=f'Rotation {df.loc[random_index, "Rotation"]}')
    axes[2].set_title(label3)
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel(label3)

    # Adding the Path as a title
    path_title = f'Path: {df.loc[random_index, "Path"]}'
    fig.suptitle(path_title, y=0.92, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_car_attributes(df):
    # Count the occurrences of each unique value in the 'Rotation' column and sort by label
    counts_translation = df['CarProfile'].value_counts().sort_index()
    counts_position = df["Velocity"].value_counts().sort_index()

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the first bar chart
    counts_translation.plot(kind='bar', ax=ax1)
    ax1.set_title('CarProfile')
    ax1.set_xlabel('CarProfile')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)  # Rotate x labels for better readability

    # Plot the second bar chart
    counts_position.plot(kind='bar', ax=ax2)
    ax2.set_title('Velocity')
    ax2.set_xlabel('Velocity')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=0)  # Rotate x labels for better readability

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()

def plot_car_attributes_onehot(df):
    # Sum the boolean values for each car profile type to get the counts
    car_profile_counts = {
        'CarProfile_FCR': df['CarProfile_FCR'].sum(),
        'CarProfile_MPV': df['CarProfile_MPV'].sum(),
        'CarProfile_RDS': df['CarProfile_RDS'].sum(),
        'CarProfile_SUV': df['CarProfile_SUV'].sum()
    }

    # Create a DataFrame from the car profile counts
    counts_translation = pd.Series(car_profile_counts)

    # Count the occurrences of each unique value in the 'Velocity' column and sort by label
    counts_position = df['Velocity'].value_counts().sort_index()

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the first bar chart
    counts_translation.plot(kind='bar', ax=ax1)
    ax1.set_title('CarProfile')
    ax1.set_xlabel('CarProfile')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)  # Rotate x labels for better readability

    # Plot the second bar chart
    counts_position.plot(kind='bar', ax=ax2)
    ax2.set_title('Velocity')
    ax2.set_xlabel('Velocity')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=0)  # Rotate x labels for better readability

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_pedestrian_attributes(df):
    # Convert rotation labels to integers if they are not already
    df['Translation'] = df['Translation'].astype(int)
    df["Rotation"] = df["Rotation"].astype(int)
        
    # Count the occurrences of each unique value in the 'Rotation' column and sort by label
    counts_translation = df['Translation'].value_counts().sort_index()
    counts_position = df["Rotation"].value_counts().sort_index()

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot the first bar chart
    counts_translation.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Translation')
    ax1.set_xlabel('Translation')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)  # Rotate x labels for better readability

    # Plot the second bar chart
    counts_position.plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('Rotation')
    ax2.set_xlabel('Rotation')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=0)  # Rotate x labels for better readability

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_hic_values(df):
    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    # Vertical lines plot for 'HIC15_max'
    axes[0].vlines(df.index, ymin=0, ymax=df['HIC15_max'], color='blue', alpha=0.5)
    axes[0].set_title('HIC15_max')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('HIC15_max')

    # Vertical lines plot for 'HIC36_max'
    axes[1].vlines(df.index, ymin=0, ymax=df['HIC36_max'], color='green', alpha=0.5)
    axes[1].set_title('HIC36_max')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('HIC36_max')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_hic15_max_binned(df):
    # Count the occurrences of each unique value in the 'HIC15_max_binned' column and sort by label
    counts_hic15 = df['HIC15_max_binned'].value_counts().sort_index()

    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the bar chart for HIC15_max_binned
    counts_hic15.plot(kind='bar', ax=ax)
    ax.set_title('HIC15_max_binned')
    ax.set_xlabel('HIC15_max_binned')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=0)  # Rotate x labels for better readability

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()



def plot_hic_value(df, value_name):
    # Create a figure and axes for the subplot
    fig, ax = plt.subplots(figsize=(8, 4))

    # Vertical lines plot for 'HIC15_max'
    ax.vlines(df.index, ymin=0, ymax=df[value_name], color='blue', alpha=0.5)
    ax.set_title(value_name)
    ax.set_xlabel('Index')
    ax.set_ylabel(value_name)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_validation_acc_values(history):
    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot training & validation accuracy values
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Model accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Model loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Test'], loc='upper left')

    # Adjust layout
    plt.tight_layout()
    plt.show()
 