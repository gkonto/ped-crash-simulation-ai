import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class DataWindow():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns # Name of the column that we wish to predict
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)} # Create a dict with the name and index of the label column. This will be used for plotting.
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)} # Create a dict with the name and index of each column. This will be used to separate the features from the target variable.
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift

            self.total_window_size = input_width + shift
            self.input_slice = slice(0, input_width) # The slice function returns a slice object that specifies how to slice a sequence.I, this case, it says that the input slice starts at 0 and ends when we reach the input_width.
            self.input_indices = np.arange(self.total_window_size)[self.input_slice] # Assign indices to the inputs. These are useful for plotting.
            
            self.label_start = self.total_window_size - self.label_width # Get the index at which the labels starts. In this case, it is the total window size minus the width of the label.
            self.labels_slice = slice(self.label_start, None) # The same steps that were applied for the inputs are applied for labels
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :] # Slice the window to get the labels using the labels_slice defined in __init__
        if self.label_columns is not None: # Is we have more than one target, we stack the labels.
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1
            )
        inputs.set_shape([None, self.input_width, None]) # The shape will be [batch, time, features]. At this point, we only specifythe time dimension and allow the batch and feature dimensions to be defined later.
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def plot(self, model=None, plot_col="traffic_volume", max_subplots=3):
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.sublot(3, 1, n + 1)
            plt.ylabel(f"{plot_col} [scaled]")
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label="Inputs", marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            
            if label_col_index is None:
                continue
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s', label="Labels", c='green', s=64)
            if n == 0:
                plt.legend()
        plt.xlabel("Time (h)")
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        df = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )
        ds = df.map(self.split_to_inputs_labels)
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def sample_batch(self):
        result = getattr(self, "_sample_batch", None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result


        
        