from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TanhMultiplier(nn.Module):
    def __init__(self) -> None:
        super(TanhMultiplier, self).__init__()
        w_init = torch.ones(size=(1,), dtype=torch.float32)
        self.multiplier = torch.autograd.Variable(w_init, requires_grad=True)
    
    def forward(self, inputs):
        exp_multiplier = torch.exp(self.multiplier)
        return torch.tanh(inputs / exp_multiplier) * exp_multiplier


class Attention(nn.Module):
    def __init__(self, dim_qk, dim_v):
        super().__init__()
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.q_net = nn.Linear(dim_qk, dim_qk)
        self.k_net = nn.Linear(dim_qk, dim_qk)
        self.v_net = nn.Linear(dim_v, dim_v)
    
    def forward(self, q, k, v):
        """
        q: (batch_size, x_dim)
        k: (batch_size, retrieval_size, x_dim)
        v: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            (batch_size, x_dim + y_dim)
        """
        discrete_flag = False
        if len(q.shape) == 3:
            discrete_flag = True
            num_dim = q.shape[1]
            num_cat = q.shape[2]
            q = q.reshape(q.shape[0], -1)
            k = k.reshape(k.shape[0], k.shape[1], -1)
            v = v.reshape(v.shape[0], v.shape[1], -1)
        q = self.q_net(q)
        k = self.k_net(k)
        v = self.v_net(v)
        scores = torch.einsum("ik,ijk->ij", q, k) / ((self.dim_qk) ** 0.5)
        weights = F.softmax(scores, dim=-1) # (batch_size, size_retrieval_set)
        aggr = (v * weights.unsqueeze(-1)).sum(dim=1)
        if discrete_flag: aggr = aggr.reshape(aggr.shape[0], num_dim + 1, num_cat)
        return aggr


class RidgeAggregator(nn.Module):
    def __init__(self, _lambda=0.1) -> None:
        super().__init__()
        self._lambda = _lambda
    
    def forward(self, q, k, v):
        """
        q: (batch_size, x_dim)
        k: (batch_size, retrieval_size, x_dim)
        v: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            (batch_size, x_dim + y_dim)
        Ridge regression:
            w * (k + λI) = q
            w = (k * k.T + λI)^(-1) * k * q
        """
        discrete_flag = False
        if len(q.shape) == 3:
            discrete_flag = True
            num_dim = q.shape[1]
            num_cat = q.shape[2]
            q = q.reshape(q.shape[0], -1)
            k = k.reshape(k.shape[0], k.shape[1], -1)
            v = v.reshape(v.shape[0], v.shape[1], -1)
        A = torch.bmm(k, k.permute(0,2,1)) \
             + self._lambda * torch.repeat_interleave(torch.eye(k.shape[1], device=device).unsqueeze(0), repeats=k.shape[0], dim=0)
        B = torch.bmm(k, q.unsqueeze(-1))
        weights = torch.bmm(torch.inverse(A), B) # (batch_size, size_retrieval_set, 1)
        # print(float(weights.mean()), "±", float(weights.std()), ", ", float(weights.min()), "~", float(weights.max()))
        weights = torch.clip(weights, min=-0.9, max=1.1)
        weights = weights.squeeze(-1)
        sum = weights.sum(dim=-1).unsqueeze(-1).repeat_interleave(repeats=weights.shape[1], dim=-1)
        weights /= (sum + 1e-8)
        weights = weights.unsqueeze(-1)
        # print(float(weights.mean()), "±", float(weights.std()), ", ", float(weights.min()), "~", float(weights.max()))
        weights = torch.clip(weights, min=-10, max=10)
        aggr = (v * weights).sum(dim=1)
        if discrete_flag: aggr = aggr.reshape(aggr.shape[0], num_dim + 1, num_cat)
        return aggr


class DistanceAggregator(nn.Module):
    def __init__(self, dim_qk, dim_v):
        super().__init__()
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.q_net = nn.Linear(dim_qk, dim_qk)
        self.k_net = nn.Linear(dim_qk, dim_qk)
        self.v_net = nn.Linear(dim_v, dim_v)
    
    def forward(self, q, k, v):
        """
        q: (batch_size, x_dim)
        k: (batch_size, retrieval_size, x_dim)
        v: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            (batch_size, x_dim + y_dim)
        """
        discrete_flag = False
        if len(q.shape) == 3:
            discrete_flag = True
            num_dim = q.shape[1]
            num_cat = q.shape[2]
            q = q.reshape(q.shape[0], -1)
            k = k.reshape(k.shape[0], k.shape[1], -1)
            v = v.reshape(v.shape[0], v.shape[1], -1)
        q = self.q_net(q)
        k = self.k_net(k)
        v = self.v_net(v)
        distance = torch.cdist(q.unsqueeze(1), k).squeeze(1)
        weights = F.softmax(1 / distance, dim=-1) # (batch_size, size_retrieval_set)
        aggr = (v * weights.unsqueeze(-1)).sum(dim=1)
        if discrete_flag: aggr = aggr.reshape(aggr.shape[0], num_dim + 1, num_cat)
        return aggr


class RIM(nn.Module):
    def __init__(self,
                 input_shape,
                 activations=(nn.ReLU, nn.ReLU),
                 hidden=2048,
                 dropout=0.0,
                 final_tanh=False,
                 aggregation_method="distance") -> None:
        super().__init__()
        self.input_dim = input_shape[0]
        self.input_cat = input_shape[1] if len(input_shape) > 1 else 1
        self.input_size = np.prod(input_shape)
        if aggregation_method == "distance":
            self.aggregator = DistanceAggregator(dim_qk=self.input_size, dim_v=self.input_size + self.input_cat)
        elif aggregation_method == "attention":
            self.aggregator = Attention(dim_qk=self.input_size, dim_v=self.input_size + self.input_cat)
        else:
            self.aggregator = RidgeAggregator(_lambda=0.1)

        self.fc_layers = build_forward_model(
            input_shape=self.input_size * 2 + self.input_cat,
            activations=activations,
            hidden=hidden,
            dropout=dropout,
            final_tanh=final_tanh
        )
    
    def forward(self, target, retrieval_set):
        """
        target: (batch_size, x_dim)
        retrieval_set: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            output/output1: (batch_size, y_dim)
        """
        key = retrieval_set[:,:,:self.input_dim]
        value = retrieval_set[:,:,:self.input_dim + 1]
        self.rl = self.aggregator(target, key, value)
        # if self.rl.requires_grad == True: self.rl.retain_grad()
        output = self.fc_layers(torch.cat([target, self.rl], dim=1))
        return output
    

class ROMO(nn.Module):
    def __init__(self,
                 input_shape,
                 activations=(nn.ReLU, nn.ReLU),
                 hidden=2048,
                 dropout=0.0,
                 final_tanh=False,
                 aggregation_method="distance",
                 weights=[0.5, 0.5]) -> None:
        super().__init__()
        self.weights =weights
        self.input_dim = input_shape[0]
        self.input_cat = input_shape[1] if len(input_shape) > 1 else 1
        self.input_size = np.prod(input_shape)
        if aggregation_method == "distance":
            self.aggregator = DistanceAggregator(dim_qk=self.input_size, dim_v=self.input_size + self.input_cat)
        elif aggregation_method == "attention":
            self.aggregator = Attention(dim_qk=self.input_size, dim_v=self.input_size + self.input_cat)
        else:
            self.aggregator = RidgeAggregator(_lambda=0.1)

        self.fc_layers = build_forward_model(
            input_shape=self.input_size * 2 + self.input_cat,
            activations=activations,
            hidden=hidden,
            dropout=dropout,
            final_tanh=final_tanh
        )
        self.fc_layers_main = build_forward_model(
            input_shape=self.input_size,
            activations=activations,
            hidden=hidden,
            dropout=dropout,
            final_tanh=final_tanh
        )
    
    def forward(self, target, retrieval_set):
        """
        target: (batch_size, x_dim)
        retrieval_set: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            output/output1: (batch_size, y_dim)
        """
        key = retrieval_set[:,:,:self.input_dim]
        value = retrieval_set[:,:,:self.input_dim + 1]
        self.rl = self.aggregator(target, key, value)
        # if self.rl.requires_grad == True: self.rl.retain_grad()
        output_0 = self.fc_layers(torch.cat([target, self.rl], dim=1))
        output_1 = self.fc_layers_main(target)
        output = output_0 * self.weights[0] + output_1 * self.weights[1]
        return output, (output_0, output_1)


class DimensionRIM(RIM):
    def __init__(self,
                 input_shape,
                 opt_channel,
                 activations=(nn.ReLU, nn.ReLU),
                 hidden=2048,
                 dropout=0,
                 final_tanh=False,
                 aggregation_method="distance",
                 weights=[0.5, 0.5]) -> None:
        super().__init__(input_shape, activations, hidden, dropout, final_tanh, aggregation_method, weights)
        self.opt_channel = np.prod(opt_channel)
        self.fc_layers = build_forward_model(
            input_shape=self.input_size * 2 + self.input_cat,
            activations=activations,
            hidden=hidden,
            dropout=dropout,
            final_tanh=final_tanh
        )
        self.fc_layers_main = build_forward_model(
            input_shape=self.opt_channel,
            activations=activations,
            hidden=hidden,
            dropout=dropout,
            final_tanh=final_tanh
        )
    
    def forward(self, target, retrieval_set):
        """
        target: (batch_size, x_dim)
        retrieval_set: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            output/output1: (batch_size, y_dim)
        """
        key = retrieval_set[:,:,:self.input_dim]
        value = retrieval_set[:,:,:self.input_dim + 1]
        rl = self.aggregator(target, key, value)
        self.input = torch.cat([target, rl], dim=1)
        if self.input.requires_grad == True: self.input.retain_grad()
        output_0 = self.fc_layers(self.input)
        output_1 = self.fc_layers_main(self.input[:, :self.opt_channel])
        output = output_0 * self.weights[0] + output_1 * self.weights[1]
        return output, (output_0, output_1)



def build_forward_model(input_shape,
                        activations=(nn.ReLU, nn.ReLU),
                        hidden=2048,
                        dropout=0.0,
                        final_tanh=False):
    """Creates a torch model that outputs a probability distribution
    specifying the score corresponding to an input x.

    Args:

    input_shape: tuple[int]
        the shape of input tensors to the model
    activations: tuple[str]
        the name of activation functions for every hidden layer
    hidden: int
        the global hidden size of the network
    dropout: float
        the probability of an element to be zeroed
    """
    
    layers = [nn.Flatten()]
    input_size = np.prod(input_shape)
    output_size = hidden

    for act in activations:
        layers.extend([nn.Linear(input_size, output_size)])
        if dropout != 0:
            layers.extend([nn.Dropout(p=dropout)])
        layers.extend([eval(act)()])
        input_size = hidden
        output_size = hidden
    layers.extend([nn.Linear(hidden, 1)])

    if final_tanh:
        layers.extend([TanhMultiplier()])

    return nn.Sequential(*layers)


def build_discriminator_model(input_shape):
    
    input_size = np.prod(input_shape)
    layers = [
        nn.Flatten(),
        nn.Linear(input_size, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    ]

    return nn.Sequential(*layers)

def build_rep_model(input_shape,
                    output_shape,
                    activations=(nn.ReLU, nn.ReLU),
                    hidden_size=2048):
    
    layers = [nn.Flatten()]
    input_size = np.prod(input_shape)
    output_size = hidden_size

    for act in activations:
        layers.extend([nn.Linear(input_size, output_size)])
        layers.extend([eval(act)()])
        input_size = hidden_size
        output_size = hidden_size
    layers.extend([nn.Linear(hidden_size, np.prod(output_shape))])

    return nn.Sequential(*layers)


# forward_model = build_forward_model((200, 5))