import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from stgcn import STGCN
from utils import generate_dataset, load_traffic_data, get_normalized_adj

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

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['font.size'] = 14

use_gpu = True
num_timesteps_input = 12
num_timesteps_output = 12

epochs = 0
batch_size = 120

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', default=use_gpu, action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    print('GPU')
else:
    print('CPU')
    args.device = torch.device('cpu')


def train_epoch(training_input, training_target, batch_size):
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(7)

    A, X, means, stds = load_traffic_data()

    # 6:2:2
    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    training_target = training_target[:, :, :12]
    val_target = val_target[:, :, :12]
    test_target = test_target[:, :, :12]

    A_wave = get_normalized_adj(A).astype(np.float32)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                12).to(device=args.device)

    training_losses = []
    validation_losses = []
    validation_maes = []
    val_epoch_best_loss = 1000000000
    test_best_mae = 1000000
    no_optim = 0
    lr = 6e-4
    NAME = 'STGCN_traffic_12step'

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_criterion = nn.MSELoss()
    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.no_grad():
            net.eval()

            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)

            val_out = net(A_wave, val_input)
            val_loss = loss_criterion(val_out, val_target).to(device="cpu")
            validation_losses.append(val_loss.detach().numpy().item())

            val_out_unnormalized = val_out.detach().cpu().numpy()*stds[0]+means[0]
            val_target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]

            mae = np.mean(np.absolute(val_out_unnormalized - val_target_unnormalized))
            validation_maes.append(mae)

            val_out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        print("Epoch: ", epoch)
        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))
        print()

        # Early stop
        #     if train_epoch_loss >= train_epoch_best_loss:
        if validation_losses[-1] >= val_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            val_epoch_best_loss = validation_losses[-1]
            #         train_epoch_best_loss = train_epoch_loss
            best_epoch = epoch
            torch.save(net, 'weights/' + NAME + '.th')

        if no_optim > 6:
            print('early stop at %d epoch' % epoch)
            break

    # Test
    net = torch.load('weights/traffic/STGCN_traffic_12step_best.th')
    test_out = net(A_wave, test_input.to(device=args.device))

    test_out_unnormalized = test_out.detach().cpu().numpy() * stds[0] + means[0]
    test_target_unnormalized = test_target.numpy() * stds[0] + means[0]
    test_out_unnormalized[test_out_unnormalized < 0] = 0

    print('————Test Accuracy Result————')
    Calculate_Acc(test_target_unnormalized, test_out_unnormalized)

    # np.save('output/STGCN_traffic_testoutput_12step.npy', test_out_unnormalized)