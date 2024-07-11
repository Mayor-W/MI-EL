#-*- coding = utf-8 -*-
#@Time: 2024/5/1 21:02
#@Author: Mayor

import math
import numpy as np
import torch
import torch.nn as nn

# Temporal embedding branch
class TemporalEmbedding(nn.Module):
    def __init__(self, TimeInfo, num_embeddings, timeembeddings_k, d_model, max_len, num_timesteps_output):
        super(TemporalEmbedding, self).__init__()
        self.TimeInfo = TimeInfo
        self.num_embeddings = num_embeddings
        self.d_model = d_model

        pe = torch.zeros(max_len, timeembeddings_k)
        for pos in range(max_len):
            for i in range(0, timeembeddings_k, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / timeembeddings_k)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / timeembeddings_k)))
        self.register_buffer('pe', pe)

        self.embedding_modules = nn.ModuleList([nn.Embedding(item, timeembeddings_k) for item in num_embeddings])
        self.linear_1 = nn.Linear((len(num_embeddings) + 1) * timeembeddings_k, 36 * d_model)
        self.linear_2 = nn.Linear(36 * d_model, num_timesteps_output * d_model)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, indices):
        # temporal attribute embedding
        time_attitude = torch.from_numpy(self.TimeInfo.iloc[indices].values[:, 1:].astype(np.float64)).type(
            torch.long)  # [month,week,day,hour..] batchsize*num_embeddings

        # temporal position embedding
        sincos_pe = self.pe[indices.numpy(), :]  # batchsize*d_model

        # Concatenate the embeddings
        inputs_embedding = torch.cat([self.embedding_modules[i](time_attitude[:, i])
                                      for i in range(len(self.num_embeddings))] + [sincos_pe],
                                     dim=-1)  # batchsize*(5*d_model)

        # two layers
        TE = self.linear_1(inputs_embedding)  # batchsize*(2*d_model)
        TE = self.drop(TE)
        TE = self.act(TE)

        TE = self.linear_2(TE)  # batchsize*(3*outputstep*d_model)
        TE = self.drop(TE)
        TE = self.act(TE)

        return TE


# Spatial embedding branch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh

def zero_diagonals(x):
    y = x.copy()
    y[np.diag_indices_from(y)] = 0

    return y

def compute_eigenmaps(adj_mx, k):
    A = adj_mx.copy()
    row, col = A.nonzero()
    A[row, col] = A[col, row] = 1  # 0/1 matrix, symmetric

    n_components = connected_components(csr_matrix(A), directed=False, return_labels=False)
    assert n_components == 1  # the graph should be connected

    n = A.shape[0]
    A = zero_diagonals(A)
    D = np.sum(A, axis=1) ** (-1 / 2)
    L = np.eye(n) - (A * D).T * D  # normalized Laplacian

    _, v = eigh(L)
    eigenmaps = v[:, 1:(k + 1)]  # eigenvectors corresponding to the k smallest non-trivial eigenvalues

    return eigenmaps

class SpatialEmbedding(nn.Module):
    def __init__(self, eigenmaps_k, d_model):
        super(SpatialEmbedding, self).__init__()
        self.linear_1 = nn.Linear(eigenmaps_k, 6 * d_model)
        self.linear_2 = nn.Linear(6 * d_model, d_model)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.05)

    def forward(self, eigenmaps):
        # two layers
        SE = self.linear_1(eigenmaps)
        SE = self.drop(SE)
        SE = self.act(SE)

        SE = self.linear_2(SE)
        SE = self.drop(SE)
        SE = self.act(SE)

        return SE


# Spatio-temporal ensemble model
class STEnsembleModel(nn.Module):
    def __init__(self, num_learners, adjMatrix, sim_matrix, eigenmaps_k, timeembeddings_k, TimeInfo, num_embeddings, max_len, d_model,
                 num_timesteps_output):
        super(STEnsembleModel, self).__init__()
        self.num_learners = num_learners
        self.num_timesteps_output = num_timesteps_output

        self.N = adjMatrix.shape[0]
        self.eigenmaps_k = eigenmaps_k  # the embedding dimension of eigenmaps
        # distance embeddings N*k
        eigenmaps_dist = torch.from_numpy(compute_eigenmaps(adjMatrix, eigenmaps_k).astype(np.float32))
        # similarity embeddings N*k
        eigenmaps_sim = torch.from_numpy(compute_eigenmaps(sim_matrix, eigenmaps_k).astype(np.float32))

        eigenmaps = torch.cat((eigenmaps_sim, eigenmaps_dist),dim=1) #N*2K
        self.register_buffer('eigenmaps', eigenmaps)  # N*2K

        self.spatial_embedding_modules = nn.ModuleList([SpatialEmbedding(2 * eigenmaps_k, d_model) for item in range(3)])

        self.T = num_timesteps_output
        self.num_embeddings = num_embeddings  # the number of temporal attributes
        self.max_len = max_len  # max time step
        self.timeembeddings_k = timeembeddings_k
        self.d_model = d_model  # the dimension of matrix decomposition
        self.temporal_embedding_modules = nn.ModuleList(
            [TemporalEmbedding(TimeInfo, num_embeddings, timeembeddings_k, d_model, max_len, num_timesteps_output) for item in range(3)])

        self.sematrix = 0
        self.tematrix = 0
        self.stWeight = 0


    def forward(self, F, Xindices):  # F batchsize*3*N*T
        # SFM N*d_model -> 3*N*d_model -> 3*N*d_model -> batchsize*3*N*d_model
        SEmatrix = torch.cat([self.spatial_embedding_modules[i](self.eigenmaps).unsqueeze(0)
                                      for i in range(self.num_learners)], dim=0)  # 3*N*d_model

        SEmatrix = SEmatrix.unsqueeze(0).expand(len(Xindices), -1, -1, -1) #batchsize*3*N*d_model

        self.sematrix = SEmatrix

        # TFM batchsize*(T*d_model) -> batchsize*T*d_model -> batchsize*3*T*d_model -> batchsize*3*d_model*T
        TEmatrix = torch.cat([self.temporal_embedding_modules[i](Xindices).view(-1, self.T, self.d_model).unsqueeze(1)
                              for i in range(self.num_learners)], dim=1)

        TEmatrix = TEmatrix.permute(0, 1, 3, 2)  # batchsize*3*d_model*T

        self.tematrix = TEmatrix

        # Spatio-temporal heterogeneous ensemble weight matrix batchsize*3*N*T
        stWeight = torch.matmul(SEmatrix.reshape(-1, self.N, self.d_model),
                                TEmatrix.reshape(-1, self.d_model, self.T)).view(-1, self.num_learners, self.N, self.T)

        self.stWeight = stWeight

        output = torch.sum(F * stWeight, dim=1)

        return output