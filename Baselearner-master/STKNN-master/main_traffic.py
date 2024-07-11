#-*- coding = utf-8 -*-
#@Time: 2024/4/3 16:18
#@Author: Mayor

import pandas as pd
import geopandas as gpd
import numpy as np

# Load monitoring station data
stationData = gpd.GeoDataFrame.from_file('data/traffic/stationTraffic.shp',encoding='gb2312')

# Load traffic flow data
TrafficData = pd.read_csv('data/traffic/traffic_regular.csv',encoding="utf-8")

# Divide the data
XData = TrafficData.values.transpose(1, 0).reshape(8064, -1, 1).transpose(1, 2, 0).astype(np.float64)
# 6:2:2
split_line1 = int(XData.shape[2] * 0.6)
split_line2 = int(XData.shape[2] * 0.8)

train_original_data = XData[:, :, :split_line1] #(56, 1, 4838)
val_original_data = XData[:, :, split_line1:split_line2] #(56, 1, 1613)
test_original_data = XData[:, :, split_line2:] #(56, 1, 1613)

# Training Validation Test data
def generate_dataset(X, num_timesteps_input, num_timesteps_output):
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


training_input = training_input.reshape(-1, 56, num_timesteps_input)
val_input = val_input.reshape(-1, 56, num_timesteps_input)
test_input = test_input.reshape(-1, 56, num_timesteps_input)


from geopy.distance import geodesic
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

positionData=stationData[['Field3','Field2']].values

# Spatial proximity matrix
distanceMatrix = np.zeros((56, 56))

for i in range(56):
    for j in range(56):
        distance = calculate_distance(positionData[i], positionData[j])
        relationIndex = np.corrcoef(train_original_data[i].reshape(-1), train_original_data[j].reshape(-1))[0, 1]
        distanceMatrix[i][j] = np.power(distance, 1-relationIndex)


def GaussianWeighted(distArray, au):
    return np.exp(-(distArray * distArray) / (4 * au * au)) / (4 * np.pi * au * au)


def ST_KNN(input, target, training_input, training_target, dist_threshold, time_threshold, K_threshold, au):
    # Result matrix
    outputResult = np.empty_like(target)  # (573, 36, 6)

    # for each test sample
    for sampleIndex in range(input.shape[0]):
        print('Test Sample:', sampleIndex + 1)
        sampleData = input[sampleIndex]  # 36*12
        for stationIndex in range(input.shape[1]):
            # select station in thresholds
            stationIndices = np.where(distanceMatrix[stationIndex] < dist_threshold)[0]

            # original spatial-temporal state matrix
            testMatrix = sampleData[stationIndices][:,-time_threshold:]  # n*t

            # historical spatial-temporal state matrix
            trainMatrix = training_input[:, stationIndices, -time_threshold:]  # 1755*n*t

            # spatial-temporal weight
            SWeight = np.diag(distanceMatrix[stationIndex][stationIndices] / np.sum(
                distanceMatrix[stationIndex][stationIndices]))  # n*n
            TWeight = np.diag(
                np.arange(1, time_threshold + 1) / np.sum(np.arange(1, time_threshold + 1)))  # t*t

            # weighted spatial-temporal state matrix
            testSTMatrix = np.dot(np.dot(SWeight, testMatrix), TWeight)  # n*t

            trainSTMatrix = np.transpose(trainMatrix, (1, 2, 0))  # n*t*1755
            trainSTMatrix = np.tensordot(SWeight, trainSTMatrix, axes=([1], [0]))  # n*n  n*t*1755 -> n*t*1755
            trainSTMatrix = np.transpose(trainSTMatrix, (2, 0, 1))  # 1755*n*12
            trainSTMatrix = np.tensordot(trainSTMatrix, TWeight, axes=([2], [0]))  # 1755*n*t  t*t -> 1755*n*t

            # calculate distance
            similarList = []
            for matrix in trainSTMatrix:  # 1755
                product = np.dot(testSTMatrix - matrix, (testSTMatrix - matrix).T).trace()
                similarList.append(product)
            similarList = np.array(similarList)  # 1755

            # select K neighbors
            neighborsIndex = np.argsort(similarList)[:K_threshold]  # K
            XNeighbors = training_target[neighborsIndex, stationIndex, :]  # (K, 6)

            Distneighbors = similarList[neighborsIndex]  # K
            Distneighbors = (Distneighbors - np.min(Distneighbors)) / (np.max(Distneighbors) - np.min(Distneighbors))

            # Gaussian weight
            Weights = GaussianWeighted(Distneighbors, au)[:, np.newaxis]  # (K,1)

            # predict
            outputResult[sampleIndex][stationIndex] = np.sum(Weights * XNeighbors, axis=0) / Weights.sum()  # (K, 6)

    return outputResult


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
    rmse = np.sqrt(mean_squared_error(label, output))
    mape = mean_absolute_percentage_error(label, output)

    print('MAE:', mae)
    print('RMSE:', rmse)
    print('MAPE:', mape)

    return mae, rmse, mape


d = 1 # spatial neighbors thresholds
t = 6 # temporal window thresholds
k = 32 # the number of neighbors
au = 0.4 # Gaussian weight thresholds
outputResult = ST_KNN(test_input, test_target, training_input, training_target, d, t, k, au)

print('——ST-KNN——')
Calculate_Acc(test_target[:,:,:3], outputResult[:,:,:3])
Calculate_Acc(test_target[:,:,:6], outputResult[:,:,:6])
Calculate_Acc(test_target, outputResult)

# np.save('output/STKNN_traffic_testoutput.npy', outputResult)