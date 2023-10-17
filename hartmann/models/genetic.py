import torch
import copy
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def UniformCrossover(parent1, parent2, p):
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    for i in range(len(parent1)):
        if torch.rand(1, device=device) < p:
            tmp = child1[i]
            child1[i] = child2[i]
            child2[i] = tmp
    return child1, child2

def NormalMutate(x, p, opt_limit):
    ub = opt_limit["x_opt_ub"].reshape(-1)
    lb = opt_limit["x_opt_lb"].reshape(-1)
    for i in range(len(x)):
        if torch.rand(1, device=device) < p:
            randn = torch.randn(1, device=device).squeeze() + 1.0
            x[i] *= randn
    return x

def UniformMutate(x, p, opt_limit):
    ub = opt_limit["x_opt_ub"].reshape(-1)
    lb = opt_limit["x_opt_lb"].reshape(-1)
    for i in range(len(x)):
        if torch.rand(1, device=device) < p:
            x[i] = torch.rand(1, device=device) * (ub[i] - lb[i]) + lb[i]
    return x