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


class ConservativeMaximumLikelihood(nn.Module):
    def __init__(self,
                 forward_model,
                 forward_model_opt=optim.Adam,
                 forward_model_lr=0.001,
                 initial_alpha=1.0,
                 alpha_opt=optim.Adam,
                 alpha_lr=0.05,
                 target_conservatism=1.0,
                 negatives_fraction=0.5,
                 lookahead_steps=50,
                 lookahead_backprop=True,
                 solver_beta=0.0,
                 solver_lr=0.01,
                 solver_interval=10,
                 solver_warmup=500,
                 solver_steps=1,
                 constraint_type="mix",
                 entropy_coefficient=0.9,
                 continuous_noise_std=0.0
                 ) -> None:
        """Build a trainer for an conservative forward model trained using
        an adversarial negative sampling procedure.

        Args:

        forward_model: nn.Module
            a torch.nn.Module that accepts a batch of designs x as input
            and predicts a batch of scalar scores as output
        forward_model_opt: torch.optim
            an optimizer that determines how the weights of the forward
            model are updated during training (eg: Adam)
        forward_model_lr: float
            the learning rate passed to the optimizer used to update the
            forward model weights (eg: 0.0003)
        initial_alpha: float
            the initial value for the lagrange multiplier, which is jointly
            optimized with the forward model during training.
        alpha_opt: torch.optim
            an optimizer that determines how the lagrange multiplier
            is updated during training (eg: Adam)
        alpha_lr: float
            the learning rate passed to the optimizer used to update the
            lagrange multiplier (eg: 0.05)
        target_conservatism: float
            the degree of overestimation that the forward model is trained
            via dual gradient ascent to have no more than
        negatives_fraction: float
            (deprecated) a deprecated parameter that should be set to 1,
            and will be phased out in future versions
        lookahead_steps: int
            the number of steps of gradient ascent used when finding
            negative samples for training the forward model
        lookahead_backprop: bool
            whether or not to allow gradients to flow back into the
            negative sampler, required for gradients to be unbiased
        solver_beta: float
            the value of beta to use during trust region optimization,
            a value as large as 0.9 typically works
        solver_lr: float
            the learning rate used to update negative samples when
            optimizing them to maximize the forward models predictions
        solver_interval: int
            the number of training steps for the forward model between
            updates to the set of solution particles
        solver_warmup: int
            the number of steps to train the forward model for before updating
            the set of solution particles for the first time
        solver_steps: int
            (deprecated) the number of steps used to update the set of
            solution particles at once, set this to 1
        constraint_type: str in ["dataset", "mix", "solution"]
            (deprecated) a deprecated parameter that should always be set
            equal to "mix" with negatives_fraction = 1.0
        continuous_noise_std: float
            standard deviation of gaussian noise added to the design variable
            x while training the forward model
        """

        super(ConservativeMaximumLikelihood, self).__init__()
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_opt(
            self.forward_model.parameters(),
            lr=forward_model_lr
        )

        # lagrangian dual descent variables
        log_alpha = np.log(initial_alpha).astype(np.float32)
        self.log_alpha = torch.tensor(log_alpha, requires_grad=True, device=device)
        self.alpha_opt = alpha_opt([self.log_alpha], lr=alpha_lr)

        # parameters for controlling the lagrangian dual descent
        self.target_conservatism = target_conservatism
        self.negatives_fraction = negatives_fraction
        self.lookahead_steps = lookahead_steps
        self.lookahead_backprop = lookahead_backprop

        # parameters for controlling learning rate for negative samples
        self.solver_lr = solver_lr
        self.solver_interval = solver_interval
        self.solver_warmup = solver_warmup
        self.solver_steps = solver_steps
        self.solver_beta = solver_beta
        self.entropy_coefficient = entropy_coefficient

        # extra parameters for controlling data noise
        self.continuous_noise_std = continuous_noise_std
        self.constraint_type = constraint_type

        # save the state of the solution found by the model
        self.step = torch.autograd.Variable(torch.tensor(0, dtype=torch.int32), device=device)
        self.solution = None
        self.particle_loss = None
        self.particle_constraint = None
        self.done = None

    def lookahead(self, x, steps):
        """Using gradient descent find adversarial versions of x that maximize
        the score predicted by the forward model

        Args:

        x: torch.Tensor
            the original value of the tensor being optimized
        steps: int
            the number of optimization steps taken

        Returns:
        
        optimized_x: torch.Tensor
            the perturbed value of x that maximizes the score function
        """
        
        # gradient ascent on the predicted score
        def gradient_step(xt):
            entropy = torch.mean((xt.unsqueeze(0) - xt.unsqueeze(1)) ** 2)
            score = (self.entropy_coefficient * entropy + self.forward_model(xt))
            score.backward(torch.ones_like(score), retain_graph=True)
            grad = xt.grad
            return xt + self.solver_lr * grad

        # use a while loop to perform gradient ascent on the score
        x.retain_grad()
        for i in range(steps):
            x = gradient_step(x)
        x.requires_grad = False

        return x


    def train_step(self, x, y):
        """Perform a training step of gradient descent on the loss function of a conservative objective model

        Args:

        x: torch.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: torch.Tensor
            a batch of training labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """
        x.requires_grad = True
        torch.add(self.step, 1)
        statistics = dict()
        batch_dim = y.shape[0]

        # corrupt the inputs with noise
        x = x + self.continuous_noise_std * torch.randn_like(x)

        # calculate the prediction error and accuracy of the model
        d = self.forward_model(x)
        mse = F.mse_loss(y, d)
        statistics[f"train/loss_mse"] = mse

        # evaluate how correct the rank of the model predictions are
        rank_corr = spearmanr(y.detach()[:, 0], d.detach()[:, 0]).correlation
        statistics[f"train/rank_corr"] = torch.tensor(rank_corr, device=device)

        # calculate negative samples starting from the dataset
        x_pos = x
        x_pos = torch.where(torch.rand([batch_dim] + [1 for _ in x.shape[1:]]) < self.negatives_fraction, x_pos, self.solution[:batch_dim])
        x_neg = self.lookahead(x_pos, self.lookahead_steps)
        if not self.lookahead_backprop:
            x_neg = x_neg.detach()

        # calculate the prediction error and accuracy of the model
        d_pos = self.forward_model(
            {"dataset": x, "mix": x_pos, "solution": self.solution[:batch_dim]}[self.constraint_type]
        )
        d_neg = self.forward_model(x_neg)
        conservatism = d_neg[:, 0] - d_pos[:, 0]
        statistics[f"train/conservatism"] = conservatism

        # build a lagrangian for dual descent
        alpha_loss = (F.softplus(self.log_alpha) * self.target_conservatism - F.softplus(self.log_alpha) * conservatism)
        statistics[f"train/alpha"] = F.softplus(self.log_alpha)

        multiplier_loss = 0.0
        if isinstance(self.forward_model[-1], TanhMultiplier):
            if self.forward_model[-1].multiplier.requires_grad == True:
                last_weight = self.forward_model[-1].multiplier.detach()
                statistics[f"train/tanh_multiplier"] = last_weight
        
        # loss that combines maximum likelihood with a constraint
        model_loss = mse + F.softplus(self.log_alpha) * conservatism + multiplier_loss
        total_loss = torch.mean(model_loss)
        alpha_loss = torch.mean(alpha_loss)

        # initialize stateful variables at the first iteration
        if self.particle_loss is None:
            initialization = torch.zeros_like(conservatism)
            self.particle_loss = torch.autograd.Variable(initialization)
            self.particle_constraint = torch.autograd.Variable(initialization)
        
        # calculate gradients using the model
        self.alpha_opt.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.forward_model_opt.zero_grad()
        total_loss.backward(retain_graph=True)

        if torch.logical_and(torch.tensor(torch.equal(self.step % self.solver_interval, torch.zeros(1))), torch.greater_equal(self.step, self.solver_warmup)):
            # take gradient steps on the model
            self.alpha_opt.step()
            self.forward_model_opt.step()

            # calculate the predicted score of the current solution
            current_score_new_model = self.forward_model(self.solution)[:, 0]

            # look into the future and evaluate future solutions
            future_new_model = self.lookahead(self.solution, self.solver_steps)
            future_score_new_model = self.forward_model(future_new_model)[:, 0]

            # evaluate the conservatism of the current solution
            particle_loss = (self.solver_beta * future_score_new_model - current_score_new_model)

            particle_loss.backward(retain_graph=True)
            solution_grad = self.solution.grad
            update = (self.solution - self.solver_lr * solution_grad)

            self.solution = torch.where(self.done, self.solution, update)
            self.particle_loss = particle_loss
            self.particle_constraint = future_score_new_model - current_score_new_model
        
        else:
            # take gradient steps on the model
            self.alpha_opt.step()
            self.forward_model_opt.step()
        
        statistics[f"train/done"] = self.done.type(torch.float32)
        statistics[f"train/particle_loss"] = self.particle_loss
        statistics[f"train/particle_constraint"] = self.particle_constraint

        return statistics

    def validate_step(self, x, y):
        """Perform a validation step on the loss function
        of a conservative objective model

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
        batch_dim = y.shape[0]
        x.requires_grad = True

        # corrupt the inputs with noise
        x = x + self.continuous_noise_std * torch.randn_like(x)

        # calculate the prediction error and accuracy of the model
        d = self.forward_model(x)
        mse = F.mse_loss(y, d)
        statistics[f"validate/loss_mse"] = mse

        # evaluate how correct the rank of the model predictions are
        rank_corr = spearmanr(y.detach()[:, 0], d.detach()[:, 0]).correlation
        statistics[f"validate/rank_corr"] = torch.tensor(rank_corr, device=device)

        # calculate negative samples starting from the dataset
        x_pos = x
        x_pos = torch.where(torch.rand([batch_dim] + [1 for _ in x.shape[1:]]) < self.negatives_fraction, x_pos, self.solution[:batch_dim])
        x_neg = self.lookahead(x_pos, self.lookahead_steps)
        if not self.lookahead_backprop:
            x_neg = x_neg.detach()
        
        # calculate the prediction error and accuracy of the model
        d_pos = self.forward_model(
            {"dataset": x, "mix": x_pos, "solution": self.solution[:batch_dim]}[self.constraint_type]
        )
        d_neg = self.forward_model(x_neg)
        conservatism = d_neg[:, 0] - d_pos[:, 0]
        statistics[f"validate/conservatism"] = conservatism

        return statistics


class TransformedMaximumLikelihood(nn.Module):
    def __init__(self,
                 forward_mdoel,
                 forward_model_optim=optim.Adam,
                 forward_model_lr=0.001,
                 logger_prefix="",
                 continuous_noise_std=0.0) -> None:
        """Build a trainer for an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        oracles: List[torch.nn.Model]
            a list of torch model that predict distributions over scores
        oracle_optim: __class__
            the optimizer class to use for optimizing the oracle model
        oracle_lr: float
            the learning rate for the oracle model optimizer
        """

        super().__init__()
        self.logger_prefix = logger_prefix
        self.continuous_noise_std = continuous_noise_std
        self.forward_model = forward_mdoel
        self.forward_model_optim = forward_model_optim(self.forward_model.parameters(), lr=forward_model_lr)
        
    
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
        x = x + self.continuous_noise_std * torch.randn_like(x)
        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d = self.forward_model(x)
        nll = F.mse_loss(y, d)
        
        # evaluate how correct the rank of the model predictions are
        rank_correlation = spearmanr(y.detach()[:, 0], d.detach()[:, 0]).correlation

        multiplier_loss = 0.0
        if isinstance(self.forward_model[-1], TanhMultiplier):
            if self.forward_model[-1].multiplier.requires_grad == True:
                last_weight = self.forward_model[-1].multiplier.detach()
                statistics[f"train/tanh_multiplier"] = last_weight
        
        # build the total loss and weight by the bootstrap
        total_loss = torch.mean(nll) + multiplier_loss

        self.forward_model_optim.zero_grad()
        total_loss.backward(retain_graph=True)
        self.forward_model_optim.step()

        statistics[f"{self.logger_prefix}/train/nll"] = nll
        statistics[f"{self.logger_prefix}/train/rank_corr"] = torch.tensor(rank_correlation, device=device)

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

        # corrupt the inputs with noise
        x = x + self.continuous_noise_std * torch.randn_like(x)
        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d = self.forward_model(x)
        nll = F.mse_loss(y, d)

        # evaluate how correct the rank of the model predictions are
        rank_correlation = spearmanr(y.detach()[:, 0],d.detach()[:, 0]).correlation

        statistics[f"{self.logger_prefix}/validate/nll"] = nll
        statistics[f"{self.logger_prefix}/validate/rank_corr"] = torch.tensor(rank_correlation, device=device)

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

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.train_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            ## statistics[name] = torch.stack(statistics[name], dim=0)
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

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            ## statistics[name] = torch.stack(statistics[name], dim=0)
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

        for e in range(epochs):
            for name, loss in self._train(train_data).items():
                logger.record(name, loss, e)
            for name, loss in self._validate(validate_data).items():
                logger.record(name, loss, e)


class ConservativeObjectiveModel(nn.Module):
    
    def __init__(self,
                 forward_model,
                 forward_model_optim=torch.optim.Adam,
                 forward_model_lr=0.001,
                 alpha=1.0,
                 alpha_optim=torch.optim.Adam,
                 alpha_lr=0.01,
                 overestimation_limit=0.5,
                 opt_limit={},
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
                if self.opt_limit.get('x_opt_channel') is not None:
                    x_opt_channel = self.opt_limit.get('x_opt_channel')
                    mask = torch.zeros_like(grad)
                    grad = torch.cat([grad[:, :x_opt_channel], mask[:, x_opt_channel:]], dim=-1)

                xt_ = xt + self.particle_lr * grad

                # clip xt
                if (self.opt_limit.get('x_opt_ub') is not None) and (self.opt_limit.get('x_opt_lb') is not None):
                    xt_ = torch.where(xt_ > self.opt_limit.get('x_opt_ub'), xt, xt_)
                    xt_ = torch.where(xt_ < self.opt_limit.get('x_opt_lb'), xt, xt_)

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
        statistics[f"train/loss_mse"] = mse.detach().unsqueeze(dim=0)

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
        statistics[f"validate/loss_mse"] = mse.detach().unsqueeze(dim=0)

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