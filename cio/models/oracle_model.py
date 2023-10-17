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


class MaximumLikelihoodModel(nn.Module):
    def __init__(self,
                 oracle,
                 oracle_optim=optim.Adam,
                 oracle_lr=0.001,
                 logger_prefix="",
                 noise_std=0.0,
                 model_dir="checkpoints",
                 model_load=False) -> None:
        """Build a trainer for one of an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        oracle: torch.nn.Model
            a list of torch model that predict distributions over scores
        oracle_optim: torch.optim __class__
            the optimizer class to use for optimizing the oracle model
        oracle_lr: float
            the learning rate for the oracle model optimizer
        logger_prefix: str
            indicate which model the current model is in the ensemble of probabilistic neural networks
        noise_std: float
            the standard deviation of the gaussian noise added to designs when training the forward model
        model_dir: str
            the directory to save or load model checkpoints
        model_load: bool
            whether to load model parameters from saved checkpoints or train the model from scratch and save the latest parameters
        """

        super().__init__()
        self.logger_prefix = logger_prefix
        self.noise_std = noise_std
        self.oracle = oracle
        self.oracle_optim = oracle_optim(self.oracle.parameters(), lr=oracle_lr, weight_decay=1e-8)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.oracle_optim, step_size=50, gamma=0.5)
        
        self.model_dir = model_dir
        self.model_load = model_load
    
    def train_step(self, x, y):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: torch.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: torch.Tensor
            a batch of training labels shaped like [batch_size, 1]
        b: torch.Tensor
            bootstrap indicators shaped like [batch_size, num_oracles]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """
        
        # corrupt the inputs with noise
        x = x + self.noise_std * torch.randn_like(x)
        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d = self.oracle(x)
        nll = F.mse_loss(y, d)
        
        # evaluate how correct the rank of the model predictions are
        rank_correlation = spearmanr(y.detach().cpu().numpy()[:, 0], d.detach().cpu().numpy()[:, 0]).correlation

        multiplier_loss = 0.0
        if isinstance(self.oracle[-1], TanhMultiplier):
            if self.oracle[-1].multiplier.requires_grad == True:
                last_weight = self.oracle[-1].multiplier.detach()
                statistics[f"{self.logger_prefix}/train/tanh_multiplier"] = last_weight
        
        # build the total loss and weight by the bootstrap
        total_loss = torch.mean(nll) + multiplier_loss

        self.oracle_optim.zero_grad()
        total_loss.backward(retain_graph=False)
        self.oracle_optim.step()

        statistics[f"{self.logger_prefix}/train/nll"] = nll.detach().unsqueeze(dim=0)
        statistics[f"{self.logger_prefix}/train/rank_corr"] = torch.tensor(rank_correlation, device=device).unsqueeze(dim=0)

        return statistics
    
    def validate_step(self, x, y):
        """Perform a validation step on an ensemble of models
        without using bootstrapping weights

        Args:

        x: torch.Tensor
            a batch of validation inputs shaped like [batch_size, channels]
        y: torch.Tensor
            a batch of validation labels shaped like [batch_size, 80]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d = self.oracle(x)
        nll = F.mse_loss(y, d)

        # evaluate how correct the rank of the model predictions are
        rank_correlation = spearmanr(y.detach().cpu().numpy()[:, 0],d.detach().cpu().numpy()[:, 0]).correlation

        statistics[f"{self.logger_prefix}/validate/nll"] = nll.detach().unsqueeze(dim=0)
        statistics[f"{self.logger_prefix}/validate/rank_corr"] = torch.tensor(rank_correlation, device=device).unsqueeze(dim=0)

        return statistics
    
    def _train(self,
              dataset):
        """Perform training using gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        dataset: torch.utils.DataLoader
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        # set the module in training mode
        self.train()

        statistics = defaultdict(list)
        for train_step, (x, aux, y) in enumerate(dataset):
            for name, tensor in self.train_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = torch.cat(statistics[name], dim=0)
        return statistics

    def _validate(self,
                 dataset):
        """Perform validation on an ensemble of models without
        using bootstrapping weights

        Args:

        dataset: torch.utils.DataLoader
            the validation dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        # set the module in evaluation mode
        self.eval()

        statistics = defaultdict(list)
        for validate_step, (x, aux, y) in enumerate(dataset):
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
                statistics[name] = torch.cat(statistics[name], dim=0)
        return statistics

    def launch(self,
               train_data,
               validate_data,
               logger,
               epochs):
        """Launch training and validation for the model for the specified
        number of epochs, and log statistics

        Args:

        train_data: torch.utils.DataLoader
            the training dataset already batched and prefetched
        validate_data: torch.utils.DataLoader
            the validation dataset already batched and prefetched
        logger: Logger
            an instance of the logger used for writing to tensor board
        epochs: int
            the number of epochs through the data sets to take
        """

        if self.model_load:
            self.load_model(epochs)
            logger.logger.info("Loaded {} at Epoch {}".format(self.logger_prefix, epochs-1))
        else:
            for e in range(epochs):
                logger.logger.info('{} Epoch [{}/{}]'.format(self.logger_prefix, e, epochs-1))
                for name, loss in self._train(train_data).items():
                    logger.record(name, loss, e)
                for name, loss in self._validate(validate_data).items():
                    logger.record(name, loss, e)
                self.lr_scheduler.step()
            self.save_model(epochs)
            logger.logger.info("Saved {} at Epoch {}".format(self.logger_prefix, epochs-1))
    
    def save_model(self, epoch):
        """
        Save the model
        :return: the saved model directory
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        model_ckpt = dict()
        model_ckpt["model_state_dict"] = self.oracle.state_dict()
        model_ckpt["epoch"] = epoch
        torch.save(model_ckpt, os.path.join(self.model_dir, f"{self.logger_prefix}_epoch{epoch}.tar"))
    
    def load_model(self, epoch):
        """
        Load the model
        """
        assert os.path.exists(os.path.join(self.model_dir, f"{self.logger_prefix}_epoch{epoch}.tar"))
        model_ckpt = torch.load(os.path.join(self.model_dir, f"{self.logger_prefix}_epoch{epoch}.tar"), map_location='cpu')
        self.oracle.load_state_dict(model_ckpt['model_state_dict'])
            

class GCNConvMaximumLikelihoodModel(nn.Module):
    def __init__(self,
                 oracle,
                 oracle_optim=optim.Adam,
                 oracle_lr=0.001,
                 logger_prefix="",
                 noise_std=0.0,
                 model_dir="checkpoints",
                 model_load=False,
                 max_batch_size=2048,
                 task=None) -> None:
        """Build a trainer for one of an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        oracle: torch.nn.Model
            a list of torch model that predict distributions over scores
        oracle_optim: torch.optim __class__
            the optimizer class to use for optimizing the oracle model
        oracle_lr: float
            the learning rate for the oracle model optimizer
        logger_prefix: str
            indicate which model the current model is in the ensemble of probabilistic neural networks
        noise_std: float
            the standard deviation of the gaussian noise added to designs when training the forward model
        model_dir: str
            the directory to save or load model checkpoints
        model_load: bool
            whether to load model parameters from saved checkpoints or train the model from scratch and save the latest parameters
        """

        super().__init__()
        self.logger_prefix = logger_prefix
        self.noise_std = noise_std
        self.oracle = oracle
        self.oracle_optim = oracle_optim(self.oracle.parameters(), lr=oracle_lr, weight_decay=1e-8)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.oracle_optim, step_size=50, gamma=0.5)
        
        self.model_dir = model_dir
        self.model_load = model_load

        self.is_normalize_x = task.is_normalize_x
        self.num_nodes, self.edge_index, self.edges_reserve, self.mu_adj, self.st_adj = self._task_info(task, max_batch_size)
    
    @staticmethod
    def _task_info(task, max_batch_size):
        num_nodes = int(task.aux.shape[-1] / 2)
        num_edges = num_nodes * (num_nodes - 1)

        # edge_index: (2, max_batch_size * num_edges)
        edge_index = torch.zeros((2, num_edges), dtype=torch.long, device=device)
        edge_cnt = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index[0][edge_cnt] = i
                    edge_index[1][edge_cnt] = j
                    edge_cnt += 1
        edge_indexes = torch.zeros((2, 0), dtype=torch.long, device=device)
        for i in range(max_batch_size):
            edge_indexes = torch.concat([edge_indexes, edge_index + i * num_nodes], dim=1)

        # edges_reserve: (num_edges,)
        arange = torch.arange(num_nodes * num_nodes, device=device)
        edges_reserve = arange[arange % (num_nodes + 1) != 0]

        # mu and std of adj_mx: (num_nodes * num_nodes,)
        mu_adj, st_adj = None, None
        if task.is_normalize_x:
            mu_adj = task.mu_x[:, num_nodes * (num_nodes + 4) : num_nodes * (2 * num_nodes + 4)]
            st_adj = task.st_x[:, num_nodes * (num_nodes + 4) : num_nodes * (2 * num_nodes + 4)]
            mu_adj, st_adj = torch.tensor(mu_adj, device=device), torch.tensor(st_adj, device=device)
        
        return num_nodes, edge_indexes, edges_reserve, mu_adj, st_adj
    
    def generate_gcn_input_list(self, x):
        batch_size = x.shape[0]
        num_nodes = int(((8 * x.shape[-1] + 121) ** 0.5 - 11) / 4)
        num_edges = num_nodes * (num_nodes - 1)

        # edge_index: (2, batch_size * num_edges)
        edge_index = self.edge_index[:, : batch_size * num_edges]

        # edge_weight: (batch_size * num_edges)
        adj_mx = x[:, num_nodes * (num_nodes + 4) : num_nodes * (2 * num_nodes + 4)]
        if self.is_normalize_x:
            adj_mx = adj_mx * self.st_adj + self.mu_adj
        edge_weight = adj_mx[:, self.edges_reserve].reshape(-1)
        edge_weight = torch.clamp(edge_weight, min=1e-5)

        # x: (batch_size * num_nodes, input_dim)
        cio = x[:, : num_nodes * num_nodes].reshape(batch_size, num_nodes, -1)
        input = x[:, num_nodes * num_nodes : num_nodes * (num_nodes + 4)].reshape(batch_size, num_nodes, -1)
        bs = x[:, num_nodes * (2 * num_nodes + 4) :].reshape(batch_size, num_nodes, -1)
        x = torch.cat([cio, input, bs], dim=-1).reshape(batch_size * num_nodes, -1)

        return x, edge_index, edge_weight, batch_size
    
    def train_step(self, x, y):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: torch.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: torch.Tensor
            a batch of training labels shaped like [batch_size, 80]
        b: torch.Tensor
            bootstrap indicators shaped like [batch_size, num_oracles]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """
        
        # corrupt the inputs with noise
        statistics = dict()
        x, edge_index, edge_weight, batch_size = self.generate_gcn_input_list(x)
        x = x + self.noise_std * torch.randn_like(x)

        # calculate the prediction error and accuracy of the model
        d = self.oracle(x, edge_index, edge_weight, batch_size)
        nll = F.mse_loss(y, d)
        
        # evaluate how correct the rank of the model predictions are
        rank_correlation = spearmanr(y.detach().cpu().numpy()[:, 0], d.detach().cpu().numpy()[:, 0]).correlation

        multiplier_loss = 0.0
        if isinstance(self.oracle[-1], TanhMultiplier):
            if self.oracle[-1].multiplier.requires_grad == True:
                last_weight = self.oracle[-1].multiplier.detach()
                statistics[f"{self.logger_prefix}/train/tanh_multiplier"] = last_weight
        
        # build the total loss and weight by the bootstrap
        total_loss = torch.mean(nll) + multiplier_loss

        self.oracle_optim.zero_grad()
        total_loss.backward(retain_graph=False)
        self.oracle_optim.step()

        statistics[f"{self.logger_prefix}/train/nll"] = nll.detach().unsqueeze(dim=0)
        statistics[f"{self.logger_prefix}/train/rank_corr"] = torch.tensor(rank_correlation, device=device).unsqueeze(dim=0)

        return statistics
    
    def validate_step(self, x, y):
        """Perform a validation step on an ensemble of models
        without using bootstrapping weights

        Args:

        x: torch.Tensor
            a batch of validation inputs shaped like [batch_size, channels]
        y: torch.Tensor
            a batch of validation labels shaped like [batch_size, 80]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()
        x, edge_index, edge_weight, batch_size = self.generate_gcn_input_list(x)

        # calculate the prediction error and accuracy of the model
        d = self.oracle(x, edge_index, edge_weight, batch_size)
        nll = F.mse_loss(y, d)

        # evaluate how correct the rank of the model predictions are
        rank_correlation = spearmanr(y.detach().cpu().numpy()[:, 0],d.detach().cpu().numpy()[:, 0]).correlation

        statistics[f"{self.logger_prefix}/validate/nll"] = nll.detach().unsqueeze(dim=0)
        statistics[f"{self.logger_prefix}/validate/rank_corr"] = torch.tensor(rank_correlation, device=device).unsqueeze(dim=0)

        return statistics
    
    def _train(self,
              dataset):
        """Perform training using gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        dataset: torch.utils.DataLoader
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        # set the module in training mode
        self.train()

        statistics = defaultdict(list)
        for train_step, (x, aux, y) in enumerate(dataset):
            for name, tensor in self.train_step(x, aux).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = torch.cat(statistics[name], dim=0)
        return statistics

    def _validate(self,
                 dataset):
        """Perform validation on an ensemble of models without
        using bootstrapping weights

        Args:

        dataset: torch.utils.DataLoader
            the validation dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        # set the module in evaluation mode
        self.eval()

        statistics = defaultdict(list)
        for validate_step, (x, aux, y) in enumerate(dataset):
            for name, tensor in self.validate_step(x, aux).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
                statistics[name] = torch.cat(statistics[name], dim=0)
        return statistics

    def launch(self,
               train_data,
               validate_data,
               logger,
               epochs):
        """Launch training and validation for the model for the specified
        number of epochs, and log statistics

        Args:

        train_data: torch.utils.DataLoader
            the training dataset already batched and prefetched
        validate_data: torch.utils.DataLoader
            the validation dataset already batched and prefetched
        logger: Logger
            an instance of the logger used for writing to tensor board
        epochs: int
            the number of epochs through the data sets to take
        """

        if self.model_load:
            self.load_model(epochs)
            logger.logger.info("Loaded {} at Epoch {}".format(self.logger_prefix, epochs-1))
        else:
            for e in range(epochs):
                logger.logger.info('{} Epoch [{}/{}]'.format(self.logger_prefix, e, epochs-1))
                for name, loss in self._train(train_data).items():
                    logger.record(name, loss, e)
                for name, loss in self._validate(validate_data).items():
                    logger.record(name, loss, e)
                self.lr_scheduler.step()
            self.save_model(epochs)
            logger.logger.info("Saved {} at Epoch {}".format(self.logger_prefix, epochs-1))
    
    def save_model(self, epoch):
        """
        Save the model
        :return: the saved model directory
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        model_ckpt = dict()
        model_ckpt["model_state_dict"] = self.oracle.state_dict()
        model_ckpt["epoch"] = epoch
        torch.save(model_ckpt, os.path.join(self.model_dir, f"{self.logger_prefix}_epoch{epoch}.tar"))
    
    def load_model(self, epoch):
        """
        Load the model
        """
        assert os.path.exists(os.path.join(self.model_dir, f"{self.logger_prefix}_epoch{epoch}.tar"))
        model_ckpt = torch.load(os.path.join(self.model_dir, f"{self.logger_prefix}_epoch{epoch}.tar"), map_location='cpu')
        self.oracle.load_state_dict(model_ckpt['model_state_dict'])


class OracleModel(nn.Module):
    """
    Ensemble well-trained oracle models as one Oracle, inference only, set the module in evaluation mode, freeze the weights
    """
    def __init__(self, oracle_models) -> None:
        super(OracleModel, self).__init__()
        self.oracle_models = oracle_models
        for t in self.oracle_models:
            t.eval()
            t.requires_grad_ = False
    
    def forward(self, inputs):
        # with torch.no_grad() will cause the torch auto-gradient module offline
        # with torch.no_grad():
        ensemble_outputs = []
        for oracle_model in self.oracle_models:
            outputs = oracle_model(inputs)
            ensemble_outputs.append(outputs)
        outputs = torch.stack(ensemble_outputs, dim=0)
    
        return torch.mean(outputs, dim=0), torch.var(outputs, dim=0)
