#-*- coding = utf-8 -*-
#@Time: 2024/4/2 9:20
#@Author: Mayor

import pandas as pd
import numpy as np

# Load traffic flow data
TrafficData = pd.read_csv('data/traffic/traffic_regular.csv',encoding="utf-8")

# Divide the data
XData=TrafficData.values.transpose(1, 0).reshape(8064, -1, 1).transpose(1, 2, 0).astype(np.float64)
# 6:2:2
split_line1 = int(XData.shape[2] * 0.6)
split_line2 = int(XData.shape[2] * 0.8)

train_original_data = XData[:, :, :split_line1] #(56, 1, 4838)
val_original_data = XData[:, :, split_line1:split_line2] #(56, 1, 1613)
test_original_data = XData[:, :, split_line2:] #(56, 1, 1613)

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

# (4815, 56, 12, 1)  (4815, 56, 12)
training_input, training_target = generate_dataset(train_original_data,
                                                   num_timesteps_input=num_timesteps_input,
                                                   num_timesteps_output=num_timesteps_output)
# (1590, 56, 12, 1) (1590, 56, 12)
val_input, val_target = generate_dataset(val_original_data,
                                         num_timesteps_input=num_timesteps_input,
                                         num_timesteps_output=num_timesteps_output)
# (1590, 56, 12, 1) (1590, 56, 12)
test_input, test_target = generate_dataset(test_original_data,
                                           num_timesteps_input=num_timesteps_input,
                                           num_timesteps_output=num_timesteps_output)

# Load adjacency matrix
adjacency_matrix = np.load('data/traffic/traffic_adj_mat_arima.npy', allow_pickle=True)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def Calculate_Acc(test_target, test_output):
    test_output = (test_output + 0.5).astype(int)

    label = test_target.reshape(-1, 1)
    output = test_output.reshape(-1, 1)

    zero_indices = np.where(label < 1)[0]

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
p = 10
q = 8
I = 1

test_output = np.empty_like(test_target) #(1590, 56, 12)
for i in range(len(test_input)):
    model = sm.STARIMA(p, q, (I,), test_input[i].reshape(56, num_timesteps_input).transpose(1, 0), adjacency_matrix)
    model.fit()
    test_output[i] = model.predict(test_input[i].reshape(56, num_timesteps_input).transpose(1, 0), num_timesteps_output).transpose(1, 0) #56*12

Calculate_Acc(test_target[:,:,:3], test_output[:,:,:3])
Calculate_Acc(test_target[:,:,:6], test_output[:,:,:6])
Calculate_Acc(test_target, test_output)

# np.save('output/STARIMA_traffic_testoutput_12step.npy', test_output)