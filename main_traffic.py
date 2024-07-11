#-*- coding = utf-8 -*-
#@Time: 2024/5/15 19:20
#@Author: Mayor

import pandas as pd
import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn

from model.MI_EL_traffic import STEnsembleModel

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


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
    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))


num_timesteps_input = 12
max_timesteps_output = 12
num_timesteps_output = 12

# (4815, 56, 12, 1)  (4815, 56, 12)
training_input, training_target = generate_dataset(train_original_data,
                                                   num_timesteps_input=num_timesteps_input,
                                                   num_timesteps_output=max_timesteps_output)
# (1590, 56, 12, 1) (1590, 56, 12)
val_input, val_target = generate_dataset(val_original_data,
                                         num_timesteps_input=num_timesteps_input,
                                         num_timesteps_output=max_timesteps_output)
# (1590, 56, 12, 1) (1590, 56, 12)
test_input, test_target = generate_dataset(test_original_data,
                                           num_timesteps_input=num_timesteps_input,
                                           num_timesteps_output=max_timesteps_output)

training_target = training_target[:, :, :num_timesteps_output]
val_target = val_target[:, :, :num_timesteps_output]
test_target = test_target[:, :, :num_timesteps_output]

# Generate the timestamp table
date_rng = pd.date_range(start='2021-03-01 00:00:00', end='2021-03-28 23:55:00', freq='5T')
TimeInfo = pd.DataFrame(date_rng, columns=['date_time'])
# Add weekday and hour
TimeInfo['weekday'] = TimeInfo['date_time'].dt.weekday  # 0-Monday,6-Sunday
TimeInfo['hour'] = TimeInfo['date_time'].dt.hour

# Generate the temporal position on validation and test dataset
val_indices = torch.from_numpy(np.arange(split_line1, split_line2 - num_timesteps_input - max_timesteps_output + 1))
test_indices = torch.from_numpy(np.arange(split_line2, TimeInfo.shape[0] - num_timesteps_input - max_timesteps_output + 1))


# Load output features from base learners (1755, 36, 6) (573, 36, 6) (573, 36, 6)
STARIMA_Ftrain3 = np.load('baselearner_features/traffic/3step/STARIMA_trainoutput_3step.npy')
STARIMA_Fval3 = np.load('baselearner_features/traffic/3step/STARIMA_valoutput_3step.npy')
STARIMA_Ftest3 = np.load('baselearner_features/traffic/3step/STARIMA_testoutput_3step.npy')
STARIMA_Ftrain6 = np.load('baselearner_features/traffic/6step/STARIMA_trainoutput_6step.npy')
STARIMA_Fval6 = np.load('baselearner_features/traffic/6step/STARIMA_valoutput_6step.npy')
STARIMA_Ftest6 = np.load('baselearner_features/traffic/6step/STARIMA_testoutput_6step.npy')
STARIMA_Ftrain12 = np.load('baselearner_features/traffic/12step/STARIMA_trainoutput_12step.npy')
STARIMA_Fval12 = np.load('baselearner_features/traffic/12step/STARIMA_valoutput_12step.npy')
STARIMA_Ftest12 = np.load('baselearner_features/traffic/12step/STARIMA_testoutput_12step.npy')

STKNN_Ftrain3 = np.load('baselearner_features/traffic/3step/STKNN_trainoutput_3step.npy')
STKNN_Fval3 = np.load('baselearner_features/traffic/3step/STKNN_valoutput_3step.npy')
STKNN_Ftest3 = np.load('baselearner_features/traffic/3step/STKNN_testoutput_3step.npy')
STKNN_Ftrain6 = np.load('baselearner_features/traffic/6step/STKNN_trainoutput_6step.npy')
STKNN_Fval6 = np.load('baselearner_features/traffic/6step/STKNN_valoutput_6step.npy')
STKNN_Ftest6 = np.load('baselearner_features/traffic/6step/STKNN_testoutput_6step.npy')
STKNN_Ftrain12 = np.load('baselearner_features/traffic/12step/STKNN_trainoutput_12step.npy')
STKNN_Fval12 = np.load('baselearner_features/traffic/12step/STKNN_valoutput_12step.npy')
STKNN_Ftest12 = np.load('baselearner_features/traffic/12step/STKNN_testoutput_12step.npy')


STGCN_Ftrain3 = np.load('baselearner_features/traffic/3step/STGCN_trainoutput_3step.npy')
STGCN_Fval3 = np.load('baselearner_features/traffic/3step/STGCN_valoutput_3step.npy')
STGCN_Ftest3 = np.load('baselearner_features/traffic/3step/STGCN_testoutput_3step.npy')
STGCN_Ftrain6 = np.load('baselearner_features/traffic/6step/STGCN_trainoutput_6step.npy')
STGCN_Fval6 = np.load('baselearner_features/traffic/6step/STGCN_valoutput_6step.npy')
STGCN_Ftest6 = np.load('baselearner_features/traffic/6step/STGCN_testoutput_6step.npy')
STGCN_Ftrain12 = np.load('baselearner_features/traffic/12step/STGCN_trainoutput_12step.npy')
STGCN_Fval12 = np.load('baselearner_features/traffic/12step/STGCN_valoutput_12step.npy')
STGCN_Ftest12 = np.load('baselearner_features/traffic/12step/STGCN_testoutput_12step.npy')

# Concatenate the output features Ftrain:Mtrain*3*36*step  Fval:Mval*3*36*step  Ftest:Mtest*3*36*step
Ftrain3 = np.concatenate((STARIMA_Ftrain3[:, np.newaxis, :, :],
                      STKNN_Ftrain3[:, np.newaxis, :, :],
                      STGCN_Ftrain3[:, np.newaxis, :, :]), axis=1)
Fval3 = np.concatenate((STARIMA_Fval3[:, np.newaxis, :, :],
                      STKNN_Fval3[:, np.newaxis, :, :],
                      STGCN_Fval3[:, np.newaxis, :, :]), axis=1)
Ftest3= np.concatenate((STARIMA_Ftest3[:, np.newaxis, :, :],
                      STKNN_Ftest3[:, np.newaxis, :, :],
                      STGCN_Ftest3[:, np.newaxis, :, :]), axis=1)
Ftrain3 = torch.from_numpy(Ftrain3)
Fval3 = torch.from_numpy(Fval3)
Ftest3 = torch.from_numpy(Ftest3)

Ftrain6 = np.concatenate((STARIMA_Ftrain6[:, np.newaxis, :, :],
                      STKNN_Ftrain6[:, np.newaxis, :, :],
                      STGCN_Ftrain6[:, np.newaxis, :, :]), axis=1)
Fval6 = np.concatenate((STARIMA_Fval6[:, np.newaxis, :, :],
                      STKNN_Fval6[:, np.newaxis, :, :],
                      STGCN_Fval6[:, np.newaxis, :, :]), axis=1)
Ftest6= np.concatenate((STARIMA_Ftest6[:, np.newaxis, :, :],
                      STKNN_Ftest6[:, np.newaxis, :, :],
                      STGCN_Ftest6[:, np.newaxis, :, :]), axis=1)
Ftrain6 = torch.from_numpy(Ftrain6)
Fval6 = torch.from_numpy(Fval6)
Ftest6 = torch.from_numpy(Ftest6)

Ftrain12 = np.concatenate((STARIMA_Ftrain12[:, np.newaxis, :, :],
                      STKNN_Ftrain12[:, np.newaxis, :, :],
                      STGCN_Ftrain12[:, np.newaxis, :, :]), axis=1)
Fval12 = np.concatenate((STARIMA_Fval12[:, np.newaxis, :, :],
                      STKNN_Fval12[:, np.newaxis, :, :],
                      STGCN_Fval12[:, np.newaxis, :, :]), axis=1)
Ftest12 = np.concatenate((STARIMA_Ftest12[:, np.newaxis, :, :],
                      STKNN_Ftest12[:, np.newaxis, :, :],
                      STGCN_Ftest12[:, np.newaxis, :, :]), axis=1)
Ftrain12 = torch.from_numpy(Ftrain12)
Fval12 = torch.from_numpy(Fval12)
Ftest12 = torch.from_numpy(Ftest12)


def train_epoch(training_input, training_target, batch_size):
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        F_batch, y_batch= training_input[indices], training_target[indices]
        F_batch = F_batch
        y_batch = y_batch

        out = net(F_batch, indices)

        loss = loss_criterion(out.reshape(-1,1), y_batch.reshape(-1,1))
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().numpy())

    return sum(epoch_training_losses) / len(epoch_training_losses)


def Calculate_Acc(test_target, test_output):

    # 输出结果四舍五入至整数辆
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


if __name__ == '__main__':
    torch.manual_seed(7)

    # Set parameters
    num_learners = 3
    adj_matrix = np.load("data/traffic/traffic_adj_mat08.npy")  # adjacency matrix based on spatial Euclidean distance
    adj_matrix = adj_matrix - np.eye(adj_matrix.shape[0])

    sim_matrix = np.load("data/traffic/traffic_sim_mat.npy")  # adjacency matrix based on the similarity of monitoring values
    sim_matrix = sim_matrix - np.eye(sim_matrix.shape[-1])

    eigenmaps_k = 8 # embedding dimensions of spatial embedding branch
    timeembeddings_k = 8 # embedding dimensions of temporal embedding branch
    num_embeddings = [7, 24]  # temporal attributes
    d_model = 16  # dimensions of hidden layers
    max_len = XData.shape[-1]  # max time step

    epochs = 40
    batch_size = 80

    net = STEnsembleModel(num_learners, adj_matrix, sim_matrix, eigenmaps_k, timeembeddings_k, TimeInfo, num_embeddings,
                          max_len, d_model, num_timesteps_output)

    training_losses = []
    validation_losses = []
    validation_maes = []
    train_epoch_best_loss = 1000000000
    no_optim = 0  # early stop
    lr = 2e-4  # learning rate
    NAME = 'MI_EL_traffic_12step'

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_criterion = nn.MSELoss()
    for epoch in range(epochs):
        loss = train_epoch(Ftrain12, training_target, batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.no_grad():
            net.eval()

            val_out = net(Fval12, val_indices)
            val_loss = loss_criterion(val_out.reshape(-1, 1), val_target.reshape(-1, 1)).to(device="cpu")
            validation_losses.append(val_loss.detach().numpy().item())

            mae = np.mean(np.absolute(val_out.detach().numpy() - val_target.detach().numpy() ))

            validation_maes.append(mae)

        print("Epoch: ", epoch)
        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))
        print()

        # Early stop
        if training_losses[-1] >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = training_losses[-1]
            torch.save(net.state_dict(), 'weights/traffic/' + NAME + '.th')

        if no_optim > 3:
            print('early stop at %d epoch' % epoch)
            break

    print('————Test Accuracy Result————')
    print('————12 step————')
    testmodel12 = STEnsembleModel(num_learners, adj_matrix, sim_matrix, eigenmaps_k, timeembeddings_k, TimeInfo,
                                  num_embeddings, max_len, d_model, 12)
    testmodel12.load_state_dict(torch.load('weights/traffic/MI_EL_traffic_12step_best.th'))
    testmodel12.eval()
    test_out12 = testmodel12(Ftest12, test_indices)
    Calculate_Acc(test_target.numpy(), test_out12.detach().numpy())
    # np.save('output/traffic/MI_EL_traffic_testoutput_12step.npy', test_out12.detach().numpy())
    # np.save('output/traffic/score/EST_EL_test_teMatrix_12step.npy', net.tematrix.detach().numpy())
    # np.save('output/traffic/score/EST_EL_test_seMatrix_12step.npy', net.sematrix.detach().numpy())

    print('————6 step————')
    testmodel6 = STEnsembleModel(num_learners, adj_matrix, sim_matrix, eigenmaps_k, timeembeddings_k, TimeInfo,
                                  num_embeddings, max_len, d_model, 6)
    testmodel6.load_state_dict(torch.load('weights/traffic/MI_EL_traffic_6step_best.th'))
    testmodel6.eval()
    test_out6 = testmodel6(Ftest6, test_indices)
    Calculate_Acc(test_target.numpy()[:, :, :6], test_out6.detach().numpy())
    # np.save('output/traffic/MI_EL_traffic_testoutput_6step.npy', test_out6.detach().numpy())

    print('————3 step————')
    testmodel3 = STEnsembleModel(num_learners, adj_matrix, sim_matrix, eigenmaps_k, timeembeddings_k, TimeInfo,
                                  num_embeddings, max_len, d_model, 3)
    testmodel3.load_state_dict(torch.load('weights/traffic/MI_EL_traffic_3step_best.th'))
    testmodel3.eval()
    test_out3 = testmodel3(Ftest3, test_indices)
    Calculate_Acc(test_target.numpy()[:, :, :3], test_out3.detach().numpy())
    # np.save('output/traffic/MI_EL_traffic_testoutput_3step.npy', test_out3.detach().numpy())