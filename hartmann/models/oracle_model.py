import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from scipy.stats import spearmanr
import pdb
import itertools
import numpy as np
import os
from models.forward_model import TanhMultiplier

torch.autograd.set_detect_anomaly(True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def HartmannOracle(x):
    n_samples = x.shape[0]
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], device=device)
    A = torch.tensor([[3.0, 10, 30],
                      [0.1, 10, 35],
                      [3.0, 10, 30],
                      [0.1, 10, 35]], device=device)
    P = torch.tensor([[3689, 1170, 2673],
                      [4699, 4387, 7470],
                      [1091, 8732, 5547],
                      [381, 5743, 8828]], device=device) * 1e-4
    x = torch.repeat_interleave(x.unsqueeze(1), repeats=4, dim=1)
    P = torch.repeat_interleave(P.unsqueeze(0), repeats=n_samples, dim=0)
    exponent = torch.einsum("ijk,jk->ij", (x - P) ** 2, A)
    y = -torch.einsum("ij,j->i", torch.exp(-exponent), alpha)
    return -y.reshape(-1, 1)