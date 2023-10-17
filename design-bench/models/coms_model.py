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


class ConservativeObjectiveModel(nn.Module):
    
    def __init__(self,
                 forward_model,
                 forward_model_optim=torch.optim.Adam,
                 forward_model_lr=0.001,
                 alpha=1.0,
                 alpha_optim=torch.optim.Adam,
                 alpha_lr=0.01,
                 overestimation_limit=0.5,
                 opt_limit=None,
                 particle_lr=0.05,
                 particle_gradient_steps=50,
                 entropy_coefficient=0.9,
                 noise_std=0.0,
                 model_dir="checkpoints/coms",
                 model_load=False):
        """A trainer class for building a conservative objective model
        by optimizing a model to make conservative predictions

        Arguments:

        forward_model: torch.nn.Module
            a torch.nn.Module model that accepts designs from an MBO dataset
            as inputs and predicts their score
        forward_model_optim: torch.optim
            an optimizer such as the Adam optimizer that defines
            how to update weights using gradients
        forward_model_lr: float
            the learning rate for the optimizer used to update the
            weights of the forward model during training
        alpha: float
            the initial value of the lagrange multiplier in the
            conservatism objective of the forward model
        alpha_optim: torch.optim
            an optimizer such as the Adam optimizer that defines
            how to update the lagrange multiplier
        alpha_lr: float
            the learning rate for the optimizer used to update the
            lagrange multiplier during training
        overestimation_limit: float
            the degree to which the predictions of the model
            overestimate the true score function
        particle_lr: float
            the learning rate for the gradient ascent optimizer
            used to find adversarial solution particles
        particle_gradient_steps: int
            the number of gradient ascent steps used to find
            adversarial solution particles
        entropy_coefficient: float
            the entropy bonus added to the loss function when updating
            solution particles with gradient ascent
        noise_std: float
            the standard deviation of the gaussian noise added to
            designs when training the forward model
        model_dir: str
            the directory to save or load model checkpoints
        model_load: bool
            whether to load model parameters from saved checkpoints or train the model from scratch and save the latest parameters
        """

        super(ConservativeObjectiveModel, self).__init__()
        self.forward_model = forward_model
        self.forward_model_optim = forward_model_optim(
            self.forward_model.parameters(),
            lr=forward_model_lr,
            # weight_decay=1e-8
        )
        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.forward_model_optim, step_size=50, gamma=0.5)

        # lagrangian dual descent variables
        log_alpha = np.log(alpha).astype(np.float32)
        self.log_alpha = torch.tensor(log_alpha, requires_grad=True, device=device)
        self.alpha_optim = alpha_optim([self.log_alpha], lr=alpha_lr)

        # algorithm hyper parameters
        self.overestimation_limit = overestimation_limit
        self.opt_limit = opt_limit
        self.particle_lr = particle_lr
        self.particle_gradient_steps = particle_gradient_steps
        self.entropy_coefficient = entropy_coefficient
        self.noise_std = noise_std

        self.model_dir = model_dir
        self.model_load = model_load

    def optimize(self, x, steps):
        """Using gradient descent find adversarial versions of x
        that maximize the conservatism of the model

        Args:

        x: torch.Tensor
            the starting point for the optimizer that will be
            updated using gradient ascent
        steps: int
            the number of gradient ascent steps to take in order to
            find x that maximizes conservatism

        Returns:

        optimized_x: torch.Tensor
            a new design found by perform gradient ascent starting
            from the initial x provided as an argument
        """

        # gradient ascent on the conservatism
        def gradient_step(xt):
            # make sure xt.requires_grad = True
            
            # shuffle the designs for calculating entorpy
            shuffled_indices = np.arange(xt.shape[0])
            np.random.shuffle(shuffled_indices)
            shuffled_xt = xt[shuffled_indices]

            # entropy using the gaussian kernel
            entropy = torch.mean((xt - shuffled_xt) ** 2)

            # the predicted score unscaled according to the forward model
            score = self.forward_model(xt)

            # the conservatism of the current set of particles
            loss = self.entropy_coefficient * entropy + score
            
            loss.backward(torch.ones_like(loss), retain_graph=True)
            grad = xt.grad

            with torch.no_grad():

                # clip grad
                if self.opt_limit is not None:
                    x_opt_channel = self.opt_limit
                    mask = torch.zeros_like(grad)
                    grad = torch.cat([grad[:, :x_opt_channel], mask[:, x_opt_channel:]], dim=1)

                xt_ = xt + self.particle_lr * grad

            return xt_
        
        # use a for loop to perform gradient ascent on the score
        for i in range(steps):
            x.requires_grad = True
            x = gradient_step(x)
            # in the i-th loop, derived x (i.e., xt_ returned by gradient_step()) is detached from old computation graph and x.requires_grad should be false
            assert x.requires_grad == False

        return x

    def train_step(self, x, y):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: torch.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: torch.Tensor
            a batch of training labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """
        # corrupt the inputs with noise
        x = x + self.noise_std * torch.randn_like(x)

        statistics = dict()
        
        # calculate the prediction error and accuracy of the model
        d_pos = self.forward_model(x)  # [bs, 1]
        mse = F.mse_loss(y, d_pos)
        statistics[f"train/mse"] = mse.detach().unsqueeze(dim=0)

        # evaluate how correct the rank of the model predictions are
        rank_corr = spearmanr(y.detach().cpu().numpy()[:, 0], d_pos.detach().cpu().numpy()[:, 0]).correlation
        statistics[f"train/rank_corr"] = torch.tensor(rank_corr, device=device).unsqueeze(dim=0)

        # calculate negative samples starting from the dataset
        x_neg = self.optimize(x, self.particle_gradient_steps)

        # calculate the prediction error and accuracy of the model
        d_neg = self.forward_model(x_neg)

        overestimation = d_neg[:, 0] - d_pos[:, 0]
        statistics[f"train/overestimation"] = overestimation.detach()

        # build a lagrangian for dual descent
        alpha_loss = (torch.exp(self.log_alpha) * self.overestimation_limit - torch.exp(self.log_alpha) * overestimation)
        statistics[f"train/alpha"] = torch.exp(self.log_alpha.detach()).unsqueeze(dim=0)

        # loss that combines maximum likelihood with a constraint
        model_loss = mse + torch.exp(self.log_alpha) * overestimation
        total_loss = torch.mean(model_loss)
        alpha_loss = torch.mean(alpha_loss)

        # calculate gradients using the model
        self.alpha_optim.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.forward_model_optim.zero_grad()
        total_loss.backward(retain_graph=True)

        # take gradient steps on the model
        self.alpha_optim.step()
        self.forward_model_optim.step()

        return statistics

    def validate_step(self, x, y):
        """Perform a validation step on an ensemble of models
        without using bootstrapping weights

        Args:

        x: torch.Tensor
            a batch of validation inputs shaped like [batch_size, channels]
        y: torch.Tensor
            a batch of validation labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """
        
        statistics = dict()
        
        # calculate the prediction error and accuracy of the model
        d_pos = self.forward_model(x)
        mse = F.mse_loss(y, d_pos)
        statistics[f"validate/mse"] = mse.detach().unsqueeze(dim=0)

        # evaluate how correct the rank of the model predictions are
        rank_corr = spearmanr(y.detach().cpu().numpy()[:, 0], d_pos.detach().cpu().numpy()[:, 0]).correlation
        statistics[f"validate/rank_corr"] = torch.tensor(rank_corr, device=device).unsqueeze(dim=0)

        # calculate negative samples starting from the dataset
        x_neg = self.optimize(x, self.particle_gradient_steps)

        # calculate the prediction error and accuracy of the model
        d_neg = self.forward_model(x_neg)

        overestimation = d_neg[:, 0] - d_pos[:, 0]
        statistics[f"validate/overestimation"] = overestimation.detach()
        return statistics
    
    def _train(self, dataset):
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
        for train_step, (x, y) in enumerate(dataset):
            # print(f"Train_step {train_step}")
            for name, tensor in self.train_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = torch.cat(statistics[name], dim=0)
        return statistics
    
    def _validate(self, dataset):
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
        for validate_step, (x, y) in enumerate(dataset):
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
            logger.logger.info("Loaded forward model at Epoch {}".format(epochs-1))
        else:
            for e in range(epochs):
                logger.logger.info('Epoch [{}/{}]'.format(e, epochs-1))
                for name, loss in self._train(train_data).items():
                    logger.record(name, loss, e)
                for name, loss in self._validate(validate_data).items():
                    logger.record(name, loss, e)
                # self.lr_scheduler.step()
            self.save_model(epochs)
            logger.logger.info("Saved forward model at Epoch {}".format(epochs-1))
    
    def save_model(self, epoch):
        """
        Save the model
        :return: the saved model directory
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        model_ckpt = dict()
        model_ckpt["model_state_dict"] = self.forward_model.state_dict()
        model_ckpt["epoch"] = epoch
        torch.save(model_ckpt, os.path.join(self.model_dir, f"forward_model_epoch{epoch}.tar"))
    
    def load_model(self, epoch):
        """
        Load the model
        """
        assert os.path.exists(os.path.join(self.model_dir, f"forward_model_epoch{epoch}.tar"))
        model_ckpt = torch.load(os.path.join(self.model_dir, f"forward_model_epoch{epoch}.tar"), map_location='cpu')
        self.forward_model.load_state_dict(model_ckpt['model_state_dict'])