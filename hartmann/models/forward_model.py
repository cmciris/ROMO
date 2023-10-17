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
        q = self.q_net(q)
        k = self.k_net(k)
        v = self.v_net(v)
        scores = torch.einsum("ik,ijk->ij", q, k) / ((self.dim_qk) ** 0.5)
        weights = F.softmax(scores, dim=-1) # (batch_size, size_retrieval_set)
        return (v * weights.unsqueeze(-1)).sum(dim=1)


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
        A = torch.bmm(k, k.permute(0,2,1)) \
             + self._lambda * torch.repeat_interleave(torch.eye(k.shape[1], device=device).unsqueeze(0), repeats=k.shape[0], dim=0)
        B = torch.bmm(k, q.unsqueeze(-1))
        weights = torch.bmm(torch.inverse(A), B) # (batch_size, size_retrieval_set, 1)
        # print(float(weights.mean()), "±", float(weights.std()), ", ", float(weights.min()), "~", float(weights.max()))
        weights = torch.clip(weights, min=-0.4, max=0.6)
        weights = weights.squeeze(-1)
        sum = weights.sum(dim=-1).unsqueeze(-1).repeat_interleave(repeats=weights.shape[1], dim=-1)
        weights /= (sum + 1e-8)
        weights = weights.unsqueeze(-1)
        # print(float(weights.mean()), "±", float(weights.std()), ", ", float(weights.min()), "~", float(weights.max()))
        weights = torch.clip(weights, min=-0.4, max=0.6)
        return (v * weights).sum(dim=1)


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
        q = self.q_net(q)
        k = self.k_net(k)
        v = self.v_net(v)
        distance = torch.cdist(q.unsqueeze(1), k).squeeze(1)
        distance = torch.clip(distance, min=1e-8)
        weights = F.softmax(1 / distance, dim=-1) # (batch_size, size_retrieval_set)
        return (v * weights.unsqueeze(-1)).sum(dim=1)


class RIM(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 activations=(nn.ReLU, nn.ReLU),
                 hidden=2048,
                 dropout=0.0,
                 final_tanh=False,
                 aggregation_method="distance") -> None:
        super().__init__()
        self.input_size = np.prod(input_shape)
        self.output_size = np.prod(output_shape)
        if aggregation_method == "distance":
            self.aggregator = DistanceAggregator(dim_qk=self.input_size, dim_v=self.input_size + self.output_size)
        elif aggregation_method == "attention":
            self.aggregator = Attention(dim_qk=self.input_size, dim_v=self.input_size + self.output_size)
        else:
            self.aggregator = RidgeAggregator(_lambda=0.1)

        self.fc_layers = build_forward_model(
            input_shape=self.input_size * 2 + self.output_size,
            output_shape=output_shape,
            activations=activations,
            hidden=hidden,
            dropout=dropout,
            final_tanh=final_tanh
        )
    
    def inference(self, target, retrieval_set, aggr_grad=True):
        """
        target: (batch_size, x_dim)
        retrieval_set: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            output/output1: (batch_size, y_dim)
        """
        key = retrieval_set[:,:,:self.input_size]
        value = retrieval_set[:,:,:self.input_size + self.output_size]
        if aggr_grad:
            self.rl = self.aggregator(target, key, value)
        else:
            with torch.no_grad():
                self.rl = self.aggregator(target, key, value)
        if self.rl.requires_grad == True: self.rl.retain_grad()
        output = self.fc_layers(torch.cat([target, self.rl], dim=-1))
        return output
    
    def forward(self, target, retrieval_set):
        """
        target: (batch_size, x_dim)
        retrieval_set: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            output/output1: (batch_size, y_dim)
        """
        key = retrieval_set[:,:,:self.input_size]
        value = retrieval_set[:,:,:self.input_size + self.output_size]
        self.rl = self.aggregator(target, key, value)
        if self.rl.requires_grad == True: self.rl.retain_grad()
        output = self.fc_layers(torch.cat([target, self.rl], dim=-1))
        return output


class ROMO(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 activations=(nn.ReLU, nn.ReLU),
                 hidden=2048,
                 dropout=0.0,
                 final_tanh=False,
                 aggregation_method="distance",
                 weights=[0.5,0.5]) -> None:
        super().__init__()
        self.weights = weights
        self.input_size = np.prod(input_shape)
        self.output_size = np.prod(output_shape)
        if aggregation_method == "distance":
            self.aggregator = DistanceAggregator(dim_qk=self.input_size, dim_v=self.input_size + self.output_size)
        elif aggregation_method == "attention":
            self.aggregator = Attention(dim_qk=self.input_size, dim_v=self.input_size + self.output_size)
        else:
            self.aggregator = RidgeAggregator(_lambda=0.1)

        self.fc_layers = build_forward_model(
            input_shape=self.input_size * 2 + self.output_size,
            output_shape=output_shape,
            activations=activations,
            hidden=hidden,
            dropout=dropout,
            final_tanh=final_tanh
        )
        self.fc_layers_main = build_forward_model(
            input_shape=self.input_size,
            output_shape=output_shape,
            activations=activations,
            hidden=hidden,
            dropout=dropout,
            final_tanh=final_tanh
        )
    
    def inference(self, target, retrieval_set, distil_grad=True, aggr_grad=True):
        """
        target: (batch_size, x_dim)
        retrieval_set: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            output/output1: (batch_size, y_dim)
        """
        key = retrieval_set[:,:,:self.input_size]
        value = retrieval_set[:,:,:self.input_size + self.output_size]
        if aggr_grad:
            self.rl = self.aggregator(target, key, value)
        else:
            with torch.no_grad():
                self.rl = self.aggregator(target, key, value)
        if self.rl.requires_grad == True: self.rl.retain_grad()
        if distil_grad:
            output_0 = self.fc_layers(torch.cat([target, self.rl], dim=-1))
        else:
            with torch.no_grad():
                output_0 = self.fc_layers(torch.cat([target, self.rl], dim=-1))
        output_1 = self.fc_layers_main(target)
        output = output_0 * self.weights[0] + output_1 * self.weights[1]
        return output, (output_0, output_1)
    
    def forward(self, target, retrieval_set):
        """
        target: (batch_size, x_dim)
        retrieval_set: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            output/output1: (batch_size, y_dim)
        """
        key = retrieval_set[:,:,:self.input_size]
        value = retrieval_set[:,:,:self.input_size + self.output_size]
        self.rl = self.aggregator(target, key, value)
        if self.rl.requires_grad == True: self.rl.retain_grad()
        output_0 = self.fc_layers(torch.cat([target, self.rl], dim=-1))
        output_1 = self.fc_layers_main(target)
        output = output_0 * self.weights[0] + output_1 * self.weights[1]
        return output, (output_0, output_1)


class DimensionROMO(ROMO):
    def __init__(self,
                 input_shape,
                 output_shape,
                 opt_channel,
                 activations=(nn.ReLU, nn.ReLU),
                 hidden=2048,
                 dropout=0,
                 final_tanh=False,
                 aggregation_method="distance",
                 weights=[0.5, 0.5]) -> None:
        super().__init__(input_shape, output_shape, activations, hidden, dropout, final_tanh, aggregation_method, weights)
        self.opt_channel = np.prod(opt_channel)
        self.fc_layers = build_forward_model(
            input_shape=self.input_size * 2 + self.output_size,
            output_shape=output_shape,
            activations=activations,
            hidden=hidden,
            dropout=dropout,
            final_tanh=final_tanh
        )
        self.fc_layers_main = build_forward_model(
            input_shape=self.opt_channel,
            output_shape=output_shape,
            activations=activations,
            hidden=hidden,
            dropout=dropout,
            final_tanh=final_tanh
        )
    
    def inference(self, target, retrieval_set, distil_grad=True, aggr_grad=True):
        """
        target: (batch_size, x_dim)
        retrieval_set: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            output/output1: (batch_size, y_dim)
        """
        key = retrieval_set[:,:,:self.input_size]
        value = retrieval_set[:,:,:self.input_size + self.output_size]
        if aggr_grad:
            rl = self.aggregator(target, key, value)
        else:
            with torch.no_grad():
                rl = self.aggregator(target, key, value)
        self.input = torch.cat([target, rl], dim=-1)
        if self.input.requires_grad == True: self.input.retain_grad()
        if distil_grad:
            output_0 = self.fc_layers(self.input)
        else:
            with torch.no_grad():
                output_0 = self.fc_layers(self.input)
        output_1 = self.fc_layers_main(self.input[:, :self.opt_channel])
        output = output_0 * self.weights[0] + output_1 * self.weights[1]
        return output, (output_0, output_1)
    
    def forward(self, target, retrieval_set):
        """
        target: (batch_size, x_dim)
        retrieval_set: (batch_size, retrieval_size, x_dim + y_dim)
        return:
            output/output1: (batch_size, y_dim)
        """
        key = retrieval_set[:,:,:self.input_size]
        value = retrieval_set[:,:,:self.input_size + self.output_size]
        rl = self.aggregator(target, key, value)
        self.input = torch.cat([target, rl], dim=-1)
        if self.input.requires_grad == True: self.input.retain_grad()
        output_0 = self.fc_layers(self.input)
        output_1 = self.fc_layers_main(self.input[:, :self.opt_channel])
        output = output_0 * self.weights[0] + output_1 * self.weights[1]
        return output, (output_0, output_1)


def build_forward_model(input_shape,
                        output_shape,
                        activations=(nn.ReLU, nn.ReLU),
                        hidden=2048,
                        dropout=0.0,
                        final_tanh=False):
    """Creates a torch model that outputs a probability distribution
    specifying the score corresponding to an input x.

    Args:

    input_shape: tuple[int]
        the shape of input tensors to the model
    input_n: int
        the number of base stations in the task
    activations: tuple[str]
        the name of activation functions for every hidden layer
    hidden: int
        the global hidden size of the network
    dropout: float
        the probability of an element to be zeroed
    max_std: float
        the upper bound of the learned standard deviation
    min_std: float
        the lower bound of the learned standard deviation
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
    layers.extend([nn.Linear(hidden, np.prod(output_shape))])

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


def build_oracle(input_shape,
                 output_shape,
                 activations=(nn.ReLU, nn.ReLU, nn.ReLU),
                 hidden=2048,
                 dropout=0.0,
                 final_tanh=False):
    """Creates a torch model that outputs a probability distribution
    specifying the auxiliary label corresponding to an input x.

    Args:

    input_shape: tuple[int]
        the shape of input tensors to the model
    input_n: int
        the number of base stations in the task
    activations: tuple[str]
        the name of activation functions for every hidden layer
    hidden: int
        the global hidden size of the network
    dropout: float
        the probability of an element to be zeroed
    max_std: float
        the upper bound of the learned standard deviation
    min_std: float
        the lower bound of the learned standard deviation
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
    layers.extend([nn.Linear(hidden, np.prod(output_shape))])

    if final_tanh:
        layers.extend([TanhMultiplier()])

    return nn.Sequential(*layers)