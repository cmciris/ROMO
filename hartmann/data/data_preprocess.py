import numpy as np
import os
import torch
import pdb
from typing import Tuple

def hartmann(x0, x1, x2):
    n_samples = x0.shape[0]
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = np.array([[3689, 1170, 2673],
                  [4699, 4387, 7470],
                  [1091, 8732, 5547],
                  [381, 5743, 8828]]) * 1e-4
    x = np.array([[x0, x1, x2]])
    x = np.transpose(x, [2,0,1])
    x = np.repeat(x, repeats=4, axis=1)
    P = np.repeat(np.array([P]), repeats=n_samples, axis=0)
    exponent = np.einsum("ijk,jk->ij", (x - P) ** 2, A)
    y = -np.einsum("ij,j->i", np.exp(-exponent), alpha)
    return -y

def create_mesh(n_points=10):
    x0 = np.random.random(n_points)
    x1 = np.random.random(n_points)
    x2 = np.random.random(n_points)
    # x0 = np.concatenate([x0, np.array([0.114614])], axis=0)
    # x1 = np.concatenate([x1, np.array([0.555649])], axis=0)
    # x2 = np.concatenate([x2, np.array([0.852547])], axis=0)
    y = hartmann(x0, x1, x2)
    return x0, x1, x2, y

def init_data(n_points=100):
    x0, x1, x2, y = create_mesh(n_points)
    data = {}
    data["x"] = np.concatenate([x0.reshape(-1, 1), x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=-1)
    data["y"] = y.reshape(-1, 1)
    return data

def main():
    data = init_data(n_points=12000)
    # print(data["x"][np.argmax(data["y"])], data["y"].max())
    print(data['x'].shape, data['y'].shape)
    np.save('./x.npy', data['x'])
    np.save('./y.npy', data['y'])

if __name__ == "__main__":
    main()