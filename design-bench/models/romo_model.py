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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RetrievalEnhancedMBO(nn.Module):
    
    def __init__(self,
                 forward_model,
                 forward_model_optim=torch.optim.Adam,
                 forward_model_lr=0.001,
                 alpha=1.0,
                 alpha_optim=torch.optim.Adam,
                 alpha_lr=0.01,
                 overestimation_limit=0.5,
                 size_retrieval_set=10,
                 opt_limit=None,
                 particle_lr=0.05,
                 entropy_coefficient=0.9,
                 noise_std=0.0,
                 mse_loss_weight=1,
                 retrieval_method="distance",
                 model_dir="checkpoints/romo",
                 model_load=False):
        """A trainer class for building a gradient ascent model
        by optimizing a model to make predictions minimize nll

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
        particle_lr: float
            the learning rate for the gradient ascent optimizer
            used to find adversarial solution particles
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

        super(RetrievalEnhancedMBO, self).__init__()
        self.forward_model = forward_model
        self.forward_model_optim = forward_model_optim(
            self.forward_model.parameters(),
            lr=forward_model_lr,
            weight_decay=1e-8
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.forward_model_optim, step_size=50, gamma=0.5)

        # lagrangian dual descent variables
        log_alpha = np.log(alpha).astype(np.float32)
        self.log_alpha = torch.tensor(log_alpha, requires_grad=True, device=device)
        self.alpha_optim = alpha_optim([self.log_alpha], lr=alpha_lr)

        # algorithm hyper parameters
        self.overestimation_limit = overestimation_limit
        self.size_retrieval_set = size_retrieval_set
        self.retrieval_method = retrieval_method
        self.mse_loss_weight = mse_loss_weight
        self.opt_limit = opt_limit
        self.particle_lr = particle_lr
        self.entropy_coefficient = entropy_coefficient
        self.noise_std = noise_std

        self.model_dir = model_dir
        self.model_load = model_load

    def optimize(self, x, steps, pool):
        """Using gradient ascent find adversarial versions of x
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

            # the predicted scaled negative variance according to the forward model
            retrieval_set = self.search_engine(xt, pool)
            score, _ = self.forward_model(xt, retrieval_set)

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


    def search_engine(self, x, pool, batch_size=128):
        """
        x: [batch_size, x_dim]
        pool_x: [pool_size, x_dim]
        pool_y: [pool_size, y_dim]
        retrieval_set: [batch_size, size_retrieval_set, x_dim + y_dim]
        """
        pool_x, pool_y = pool[0], pool[1]
        num_batches = (pool_x.shape[0] // batch_size) + 1
        discrete_flag = False
        if len(x.shape) == 3:
            discrete_flag = True
            num_dim, num_cat = x.shape[1], x.shape[2]
            x = x.reshape(x.shape[0], -1)
            pool_x = pool_x.reshape(pool_x.shape[0], -1)
        if self.retrieval_method == "distance":
            distance = []
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                batch_pool_x = pool_x[start:end]
                batch_distances = torch.cdist(x, batch_pool_x)
                distance.append(batch_distances)
            distance = torch.cat(distance, dim=1)
            _, indices = torch.sort(distance, dim=-1, descending=False)
        elif self.retrieval_method == "cosine":
            similarities = []
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                batch_pool_x = pool_x[start:end]
                batch_similarities = F.cosine_similarity(x.unsqueeze(1), batch_pool_x.unsqueeze(0), dim=2)
                similarities.append(batch_similarities)
            similarities = torch.cat(similarities, dim=1)
            _, indices = torch.sort(similarities, dim=-1, descending=True)
        else:
            inners = []
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                batch_pool_x = pool_x[start:end]
                batch_inner = torch.einsum("ik,jk->ij", x, batch_pool_x)
                inners.append(batch_inner)
            inners = torch.cat(inners, dim=1)
            _, indices = torch.sort(inners, dim=-1, descending=True)
        indices = indices[:, :self.size_retrieval_set]
        retrieval_x = pool_x[indices]
        retrieval_y = pool_y[indices]
        if discrete_flag:
            retrieval_x = retrieval_x.reshape(retrieval_x.shape[0], retrieval_x.shape[1], num_dim, num_cat)
            retrieval_y = retrieval_y.unsqueeze(-1).repeat_interleave(repeats=num_cat, dim=-1)
        retrieval_set = torch.cat([retrieval_x, retrieval_y], dim=2)
        return retrieval_set


    def train_step(self, x, y, pool):
        """Perform a training step of gradient descent

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

        retrieval_set = self.search_engine(x, pool)

        statistics = dict()
        
        # calculate the prediction error and accuracy of the model
        # output_0: f(x, R(x))
        # output_1: f(x)
        d, (output_0, output_1) = self.forward_model(x, retrieval_set)  # [batch_size, output_dim]
        nll = F.mse_loss(y, d)
        statistics[f"train/nll"] = nll.detach().unsqueeze(dim=0)

        # evaluate how correct the rank of the model predictions are
        rank_corr = spearmanr(y.detach().cpu().numpy()[:, 0], d.detach().cpu().numpy()[:, 0]).correlation
        statistics[f"train/rank_corr"] = torch.tensor(rank_corr, device=device).unsqueeze(dim=0)

        multiplier_loss = 0.0
        if isinstance(self.forward_model.fc_layers[-1], TanhMultiplier):
            if self.forward_model.fc_layers[-1].multiplier.requires_grad == True:
                last_weight = self.forward_model.fc_layers[-1].multiplier.detach()
                statistics[f"train/tanh_multiplier"] = last_weight
        
        overestimation = output_1[:, 0] - output_0[:, 0]
        statistics[f"train/overestimation"] = overestimation.detach()
        
        # build a lagrangian for dual descent
        alpha_loss = (torch.exp(self.log_alpha) * self.overestimation_limit - torch.exp(self.log_alpha) * overestimation)
        statistics[f"train/alpha"] = torch.exp(self.log_alpha.detach()).unsqueeze(dim=0)
        
        # build the total loss and weight by the bootstrap
        model_loss = nll + F.mse_loss(output_1, output_0) * self.mse_loss_weight + torch.exp(self.log_alpha) * overestimation
        total_loss = torch.mean(model_loss) + multiplier_loss
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
    

    def validate_step(self, x, y, pool):
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
        retrieval_set = self.search_engine(x, pool)
        
        # calculate the prediction error and accuracy of the model
        d, _ = self.forward_model(x, retrieval_set)  # [batch_size, output_dim]
        nll = F.mse_loss(y, d)
        statistics[f"validate/nll"] = nll.detach().unsqueeze(dim=0)

        # evaluate how correct the rank of the model predictions are
        rank_corr = spearmanr(y.detach().cpu().numpy()[:, 0], d.detach().cpu().numpy()[:, 0]).correlation
        statistics[f"validate/rank_corr"] = torch.tensor(rank_corr, device=device).unsqueeze(dim=0)

        return statistics
    
    
    def _train(self, dataset, pool):
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
            for name, tensor in self.train_step(x, y, pool).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = torch.cat(statistics[name], dim=0)
        return statistics
    
    def _validate(self, dataset, pool):
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
            for name, tensor in self.validate_step(x, y, pool).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = torch.cat(statistics[name], dim=0)
        return statistics
    
    def launch(self,
               train_data,
               validate_data,
               pool_data,
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
                for name, loss in self._train(train_data, pool_data).items():
                    logger.record(name, loss, e)
                for name, loss in self._validate(validate_data, pool_data).items():
                    logger.record(name, loss, e)
                self.lr_scheduler.step()
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