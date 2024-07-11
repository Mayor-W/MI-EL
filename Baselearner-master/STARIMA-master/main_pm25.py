#-*- coding = utf-8 -*-
#@Time: 2024/3/17 15:04
#@Author: Mayor

import pandas as pd
import numpy as np

# Load PM25 data
PM25Data = pd.read_csv('data/pm25/pm25_regular.csv',encoding="utf-8")

# Divide the data
XData=PM25Data.values.transpose(1, 0).reshape(2952, -1, 1).transpose(1, 2, 0)
# 6:2:2
split_line1 = int(XData.shape[2] * 0.6) + 1
split_line2 = int(XData.shape[2] * 0.8) + 1

train_original_data = XData[:, :, :split_line1] #(36, 1, 1772)
val_original_data = XData[:, :, split_line1:split_line2] #(36, 1, 590)
test_original_data = XData[:, :, split_line2:] #(36, 1, 590)

# Training Validation Test data
def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])
    return np.array(features), np.array(target)

num_timesteps_input=12
num_timesteps_output=12

# (1755, 36, 12, 1)  (1755, 36, 6)
training_input, training_target = generate_dataset(train_original_data,
                                                   num_timesteps_input=num_timesteps_input,
                                                   num_timesteps_output=num_timesteps_output)
# (567, 36, 12, 1) (567, 36, 6)
val_input, val_target = generate_dataset(val_original_data,
                                         num_timesteps_input=num_timesteps_input,
                                         num_timesteps_output=num_timesteps_output)
# (567, 36, 12, 1) (567, 36, 6)
test_input, test_target = generate_dataset(test_original_data,
                                           num_timesteps_input=num_timesteps_input,
                                           num_timesteps_output=num_timesteps_output)

# Load adjacency matrix
adjacency_matrix = np.load('data/pm25/pm25_adj_mat_arima.npy', allow_pickle=True)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def Calculate_Acc(test_target, test_output):
    label = test_target.reshape(-1, 1)
    output = test_output.reshape(-1, 1)

    zero_indices = np.where(label < 5)[0]

    label = np.delete(label, zero_indices)
    output = np.delete(output, zero_indices)

    mae = mean_absolute_error(label, output)
    mse = mean_squared_error(label, output)
    mape = mean_absolute_percentage_error(label, output)

    print('MAE:', mae)
    print('RMSE:', np.sqrt(mse))
    print('MAPE:', mape)


# STARIMA
from .pySTARMA import starma_model as sm
p = 9
q = 4
I = 1

test_output = np.empty_like(test_target) #(573, 36, 6)
for i in range(len(test_input)):
    model = sm.STARIMA(p, q, (I,), test_input[i].reshape(36, num_timesteps_input).transpose(1, 0), adjacency_matrix)
    model.fit()
    test_output[i] = model.predict(test_input[i].reshape(36, num_timesteps_input).transpose(1, 0), num_timesteps_output).transpose(1, 0) #36*6

Calculate_Acc(test_target[:,:,:3], test_output[:,:,:3])
Calculate_Acc(test_target[:,:,:6], test_output[:,:,:6])
Calculate_Acc(test_target, test_output)

# np.save('output/STARIMA_pm25_testoutput_12step.npy', test_output)
