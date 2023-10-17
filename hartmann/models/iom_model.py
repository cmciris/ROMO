import os
import sys

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


class IOMModel(nn.Module):
    def __init__(self, g, discriminator_model, mmd_param, rep_model, rep_model_lr,
                 forward_model,
                 rep_model_opt=optim.Adam,
                 forward_model_opt=optim.Adam,
                 forward_model_lr=0.001,
                 alpha=1.0,
                 alpha_opt=optim.Adam,
                 alpha_lr=0.01,
                 overestimation_limit=0.5,
                 opt_limit={},
                 particle_lr=0.05,
                 particle_gradient_steps=50,
                 entropy_coefficient=0.9,
                 noise_std=0.0,
                 oracle=None,
                 task=None,
                 model_dir='checkpoints/iom',
                 model_load=False) -> None:
        
        super(IOMModel, self).__init__()
        discriminator_model_opt = optim.Adam
        self.adversarial_loss = F.binary_cross_entropy
        self.discriminator_model = discriminator_model
        self.mmd_param = mmd_param
        self.rep_model = rep_model
        self.rep_model_lr = rep_model_lr
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_opt(self.forward_model.parameters(), lr=forward_model_lr)
        self.rep_model_opt = rep_model_opt(self.rep_model.parameters(), lr=rep_model_lr)
        self.discriminator_model_opt = discriminator_model_opt(self.discriminator_model.parameters(), lr=0.001, betas=(0.5, 0.999))

        log_alpha = np.log(alpha).astype(np.float32)
        self.log_alpha = torch.tensor(log_alpha, requires_grad=True, device=device)
        self.alpha_opt = alpha_opt([self.log_alpha], lr=alpha_lr)

        self.overestimation_limit = overestimation_limit
        self.opt_limit = opt_limit
        self.particle_lr = particle_lr
        self.particle_gradient_steps = particle_gradient_steps
        self.entropy_coefficient = entropy_coefficient
        self.noise_std = noise_std

        self.new_sample_size = 128

        self.model_dir = model_dir
        self.model_load = model_load

        # one fixed set of x that does gradient descent every epoch
        # self.g = g.clone().detach().requires_grad_(True)
        self.g = g.clone().detach()

        # initial state of this set of x
        # self.g0 = g.clone().detach().requires_grad_(True)
        self.g0 = g.clone().detach()

        self.epoch = 0
        self.oracle = oracle
        self.task = task

    def optimize(self, x, steps, **kwargs):
        # gradient ascent on the conservatism
        def gradient_step(xt):

            # shuffle the designs for calculating entropy
            shuffled_indices = np.arange(xt.shape[0])
            np.random.shuffle(shuffled_indices)
            shuffled_xt = xt[shuffled_indices]

            # entropy using the guassian kernel
            entropy = torch.mean((xt - shuffled_xt) ** 2)

            # the predicted score according to the forward model
            xt_rep = self.rep_model(xt)
            xt_rep = xt_rep/(torch.sqrt(torch.sum(xt_rep**2, dim=-1, keepdim=True) + 1e-6) + 1e-6)
            
            score = self.forward_model(xt_rep, **kwargs)

            # the conservatism of the current set of particles
            loss = self.entropy_coefficient * entropy + score

            loss.backward(torch.ones_like(loss), retain_graph=False)
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
            assert x.requires_grad == False
        
        return x

    def train_step(self, x, y):

        # corrupt the inputs with noise
        x = x + self.noise_std * torch.randn_like(x)

        statistics = dict()

        alpha_param = torch.exp(self.log_alpha)

        # pass x to the representation network and normalize rep(x)
        rep_x = self.rep_model(x)
        rep_x = rep_x/(torch.sqrt(torch.sum(rep_x**2, dim=-1, keepdim=True) + 1e-6) + 1e-6)

        # mean absolute error between y and d_pos_rep
        d_pos_rep = self.forward_model(rep_x)
        mse = F.mse_loss(y, d_pos_rep)
        statistics[f'train/mse_L2'] = mse.detach().unsqueeze(dim=0)
        mse_l1 = F.l1_loss(y, d_pos_rep)
        statistics[f'train/mse_L1'] = mse_l1.detach().unsqueeze(dim=0)

        # evaluate how correct the rank of the model predictions are
        rank_corr = spearmanr(y.detach().cpu().numpy()[:, 0], d_pos_rep.detach().cpu().numpy()[:, 0]).correlation
        statistics[f"train/rank_corr"] = torch.tensor(rank_corr, device=device).unsqueeze(dim=0)

        # calculate negative samples starting form the dataset
        x_neg = self.optimize(self.g, 1)
        self.g = x_neg.clone().detach()

        # log the task score for this set of x every 50 epochs
        if (self.epoch % 50 == 0):
            with torch.no_grad():
                score_here = self.oracle(self.g)
            if self.task.is_normalize_y:
                score_here = self.task.denormalize_y(score_here)
            statistics['train/score_g_max'] = score_here
        
        statistics['train/distance_from_start'] = torch.mean(torch.linalg.norm(self.g - self.g0, dim=-1)).unsqueeze(dim=0)
        x_neg = x_neg[:x.shape[0]]
        
        # calculate the prediction error and accuracy of the model
        rep_x_neg = self.rep_model(x_neg)
        rep_x_neg = rep_x_neg/(torch.sqrt(torch.sum(rep_x_neg**2, dim=-1, keepdim=True) + 1e-6) + 1e-6)

        d_neg_rep = self.forward_model(rep_x_neg)

        overestimation = d_neg_rep[:,0] - d_pos_rep[:,0]
        statistics[f'train/overestimation'] = overestimation.detach()
        statistics[f'train/prediction'] = d_neg_rep.detach()

        # build a lagrangian for dual descent
        alpha_loss = (torch.exp(self.log_alpha) * self.overestimation_limit - torch.exp(self.log_alpha) * overestimation)
        statistics[f'train/alpha'] = torch.exp(self.log_alpha.detach()).unsqueeze(dim=0)

        rep_x = self.rep_model(x)
        rep_x = rep_x/(torch.sqrt(torch.sum(rep_x**2, dim=-1, keepdim=True) + 1e-6) + 1e-6)

        logged_rep = torch.mean(rep_x, dim=0)
        learned_rep = torch.mean(rep_x_neg, dim=0)
        mmd = torch.mean(F.mse_loss(learned_rep, logged_rep))
        statistics[f'train/mmd'] = mmd.detach().unsqueeze(dim=0)

        logged = torch.mean(x, dim=0)
        learned = torch.mean(x_neg, dim=0)
        mmd_before_rep = torch.mean(F.mse_loss(learned, logged))
        statistics[f'train/distance_before_rep'] = mmd_before_rep.detach().unsqueeze(dim=0)

        # gan loss
        output_shape = rep_x.shape
        valid = torch.ones([output_shape[0]], dtype=torch.float32, device=device)
        fake = torch.zeros([output_shape[0]], dtype=torch.float32, device=device)
        dis_rep_x = torch.reshape(self.discriminator_model(rep_x.detach()), [output_shape[0]])
        dis_rep_x_neg = torch.reshape(self.discriminator_model(rep_x_neg), [output_shape[0]])

        real_loss = torch.mean(torch.square(valid - dis_rep_x))
        fake_loss = torch.mean(torch.square(fake - dis_rep_x_neg))
        d_loss = (real_loss + fake_loss) / 2
        statistics[f'train/d_loss'] = d_loss.detach().unsqueeze(dim=0)
        statistics[f'train/real_loss'] = real_loss.detach().unsqueeze(dim=0)
        statistics[f'train/fake_loss'] = fake_loss.detach().unsqueeze(dim=0)

        statistics[f'train/square_dif_x_neg'] = torch.mean(torch.square(rep_x - rep_x_neg)).unsqueeze(dim=0)

        truth_pos = (torch.where(torch.transpose(self.discriminator_model(rep_x), 0, 1) >= 0.5))[1].shape[0]/rep_x.shape[0]
        statistics[f'train/accuracy_real'] = torch.tensor(truth_pos, device=device).unsqueeze(dim=0)

        truth_pos = (torch.where(torch.transpose(self.discriminator_model(rep_x_neg), 0, 1) < 0.5))[1].shape[0]/rep_x_neg.shape[0]
        statistics[f'train/accuracy_fake'] = torch.tensor(truth_pos, device=device).unsqueeze(dim=0)

        mmd_param = self.mmd_param

        model_loss1 = mse - d_loss * mmd_param
        total_loss1 = torch.mean(model_loss1)
        statistics[f'train/loss1'] = total_loss1.detach().unsqueeze(dim=0)
        alpha_loss = torch.mean(alpha_loss)

        model_loss2 = mse - d_loss * mmd_param
        total_loss2 = torch.mean(model_loss2)
        statistics[f'train/loss2'] = total_loss2.detach().unsqueeze(dim=0)

        self.alpha_opt.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.forward_model_opt.zero_grad()
        total_loss1.backward(retain_graph=True)
        self.rep_model_opt.zero_grad()
        total_loss2.backward(retain_graph=True)
        self.discriminator_model_opt.zero_grad()
        d_loss.backward(retain_graph=True)

        self.alpha_opt.step()
        self.forward_model_opt.step()
        self.rep_model_opt.step()
        self.discriminator_model_opt.step()

        return statistics


    def validate_step(self, x, y):

        statistics = dict()

        alpha_param = torch.exp(self.log_alpha)

        rep_x = self.rep_model(x)
        rep_x = rep_x/(torch.sqrt(torch.sum(rep_x**2, dim=-1, keepdim=True) + 1e-6) + 1e-6)
        # mse loss
        d_pos_rep = self.forward_model(rep_x)
        mse = F.mse_loss(y, d_pos_rep)
        statistics[f'validate/mse'] = mse.detach().unsqueeze(dim=0)

        rank_corr = spearmanr(y.detach().cpu().numpy()[:, 0], d_pos_rep.detach().cpu().numpy()[:, 0]).correlation
        statistics[f"validate/rank_corr"] = torch.tensor(rank_corr, device=device).unsqueeze(dim=0)

        x_neg = self.g
        x_neg = x_neg[:x.shape[0]]

        statistics['validate/distance_from_start'] = torch.mean(torch.linalg.norm(self.g - self.g0, dim=-1)).unsqueeze(dim=0)
        x_neg = x_neg[:x.shape[0]]
        rep_x_neg = self.rep_model(x_neg)
        rep_x_neg = rep_x_neg/(torch.sqrt(torch.sum(rep_x_neg**2, dim=-1, keepdim=True) + 1e-6) + 1e-6)

        d_neg_rep = self.forward_model(rep_x_neg)

        overestimation = d_neg_rep[:,0] - d_pos_rep[:,0]
        statistics[f'validate/overestimation'] = overestimation.detach()
        statistics[f'validate/prediction'] = d_neg_rep.detach()

        alpha_loss = (torch.exp(self.log_alpha) * self.overestimation_limit - torch.exp(self.log_alpha) * overestimation)
        statistics[f'validate/alpha'] = torch.exp(self.log_alpha.detach()).unsqueeze(dim=0)

        rep_x = self.rep_model(x)
        rep_x = rep_x/(torch.sqrt(torch.sum(rep_x**2, dim=-1, keepdim=True) + 1e-6) + 1e-6)

        logged_rep = torch.mean(rep_x, dim=0)
        learned_rep = torch.mean(rep_x_neg, dim=0)
        mmd = torch.mean(F.mse_loss(learned_rep, logged_rep))
        statistics[f'validate/mmd'] = mmd.detach().unsqueeze(dim=0)

        logged = torch.mean(x, dim=0)
        learned = torch.mean(x_neg, dim=0)
        mmd_before_rep = torch.mean(F.mse_loss(learned, logged))
        statistics[f'validate/distance_before_rep'] = mmd_before_rep.detach().unsqueeze(dim=0)

        # gan loss
        output_shape = rep_x.shape
        valid = torch.ones([output_shape[0]], dtype=torch.float32, device=device)
        fake = torch.zeros([output_shape[0]], dtype=torch.float32, device=device)
        dis_rep_x = torch.reshape(self.discriminator_model(rep_x.detach()), [output_shape[0]])
        dis_rep_x_neg = torch.reshape(self.discriminator_model(rep_x_neg), [output_shape[0]])

        real_loss = torch.mean(torch.square(valid - dis_rep_x))
        fake_loss = torch.mean(torch.square(fake - dis_rep_x_neg))
        d_loss = (real_loss + fake_loss) / 2
        statistics[f'validate/d_loss'] = d_loss.detach().unsqueeze(dim=0)
        statistics[f'validate/real_loss'] = real_loss.detach().unsqueeze(dim=0)
        statistics[f'validate/fake_loss'] = fake_loss.detach().unsqueeze(dim=0)

        statistics[f'validate/square_dif_x_neg'] = torch.mean(torch.square(rep_x - rep_x_neg)).unsqueeze(dim=0)

        truth_pos = (torch.where(torch.transpose(self.discriminator_model(rep_x), 0, 1) >= 0.5))[1].shape[0]/rep_x.shape[0]
        statistics[f'validate/accuracy_real'] = torch.tensor(truth_pos, device=device).unsqueeze(dim=0)

        truth_pos = (torch.where(torch.transpose(self.discriminator_model(rep_x_neg), 0, 1) < 0.5))[1].shape[0]/rep_x_neg.shape[0]
        statistics[f'validate/accuracy_fake'] = torch.tensor(truth_pos, device=device).unsqueeze(dim=0)

        mmd_param = self.mmd_param

        model_loss1 = mse - d_loss * mmd_param
        total_loss1 = torch.mean(model_loss1)
        statistics[f'validate/loss1'] = total_loss1.detach().unsqueeze(dim=0)
        alpha_loss = torch.mean(alpha_loss)

        model_loss2 = mse - d_loss * mmd_param
        total_loss2 = torch.mean(model_loss2)
        statistics[f'validate/loss2'] = total_loss2.detach().unsqueeze(dim=0)

        return statistics
    

    def _train(self, dataset):

        self.train()

        statistics = defaultdict(list)
        for train_step, (x, y) in enumerate(dataset):
            for name, tensor in self.train_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = torch.cat(statistics[name], dim=0)
        return statistics

    
    def _validate(self, dataset):

        self.eval()

        statistics = defaultdict(list)
        for validate_step, (x, y) in enumerate(dataset):
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = torch.cat(statistics[name], dim=0)
        return statistics
    

    def launch(self, train_data, validate_data, logger, epochs):

        if self.model_load:
            self.load_model(epochs)
            logger.logger.info('Loaded models at Epoch {}'.format(epochs-1))
        else:
            for e in range(epochs):
                logger.logger.info('Epoch [{}/{}]'.format(e, epochs-1))
                for name, loss in self._train(train_data).items():
                    logger.record(name, loss, e)
                for name, loss in self._validate(validate_data).items():
                    logger.record(name, loss, e)
            self.save_model(epochs)
            logger.logger.info('Saved models at Epoch {}'.format(epochs-1))


    def save_model(self, epoch):
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_ckpt = dict()
        model_ckpt['discriminator_model_state_dict'] = self.discriminator_model.state_dict()
        model_ckpt['rep_model_state_dict'] = self.rep_model.state_dict()
        model_ckpt['forward_model_state_dict'] = self.forward_model.state_dict()
        model_ckpt['epoch'] = epoch
        torch.save(model_ckpt, os.path.join(self.model_dir, f"models_epoch{epoch}.tar"))


    def load_model(self, epoch):
        
        assert os.path.exists(os.path.join(self.model_dir, f"models_epoch{epoch}.tar"))
        model_ckpt = torch.load(os.path.join(self.model_dir, f'models_epoch{epoch}.tar'), map_location='cpu')
        self.discriminator_model.load_state_dict(model_ckpt['discriminator_model_state_dict'])
        self.rep_model.load_state_dict(model_ckpt['rep_model_state_dict'])
        self.forward_model.load_state_dict(model_ckpt['forward_model_state_dict'])