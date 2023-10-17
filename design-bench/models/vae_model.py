import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List
from collections import defaultdict
import pdb
import numpy as np
import os

torch.autograd.set_detect_anomaly(True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """

    DEFINE AN ENCODER MODEL THAT DOWNSAMPLES
        
    """
    def __init__(self,
                 task,
                 hidden_size,
                 latent_size,
                 activation=nn.ReLU,
                 kernel_size=3,
                 num_blocks=4) -> None:
        super(Encoder, self).__init__()

        self.task = task
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.activation = activation
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks

        self.input_shape = task.input_shape
        self.shape_before_flat = [self.input_shape[0] // (2 ** (num_blocks - 1)), hidden_size]
    
    def forward(self, inputs):
        # build a model with an input layer and optional embedding
        x = nn.Embedding(num_embeddings=self.task.num_classes, embedding_dim=self.hidden_size)(inputs)  # N * L * C

        # the exponent of a positional embedding
        inverse_frequency = 1.0 / (10000.0 ** (torch.arange(0.0, self.hidden_size, 2.0) / self.hidden_size)).unsqueeze(dim=0)

        # calculate a positional embedding to break symmetry
        pos = torch.arange(0.0, x.shape[1], 1.0).unsqueeze(dim=-1)
        positional_embedding = torch.cat([torch.sin(pos * inverse_frequency), torch.cos(pos * inverse_frequency)], dim=1).unsqueeze(dim=0)

        # add the positional encoding
        x = torch.add(x, positional_embedding)  # N * L * C
        x = nn.LayerNorm(normalized_shape=x.shape[1:])(x)  # N * L * C

        x = torch.transpose(x, 1, 2)  # N * C * L

        # add several residual blocks to the model
        for i in range(self.num_blocks):

            if i > 0:
                # downsample the input sequence by 2
                # padding = "same"
                # Lout = (Lin + 2 * padding - kernel_size) / stride + 1, Lin should be even
                x = nn.AvgPool1d(kernel_size=2, padding=(((x.shape[-1] - 1) * 2 + 2 - x.shape[-1]) / 2))  # N * C * L

            # first convolution layer in a residual block
            h = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, padding='same')(x)  # N * C * L
            h = nn.LayerNorm(normalized_shape=h.shape[1:])(h)
            h = eval(self.activation)()(h)
            
            # second convolution layer in a residual block
            h = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, padding='same')(h)
            h = nn.LayerNorm(normalized_shape=h.shape[1:])(h)
            h = eval(self.activation)()(h)

            # add a residual connection to the model
            x = torch.add(x, h)

        # flatten the result and predict the params of a gaussian
        x = torch.transpose(x, 1, 2)  # N * L * C
        flattened_x = nn.Flatten()(x)
        latent_mean = nn.Linear(flattened_x.shape[1], self.latent_size)(flattened_x)
        latent_standard_dev = nn.Linear(flattened_x.shape[1], self.latent_size)(flattened_x)
        latent_standard_dev = torch.exp(latent_standard_dev)

        return latent_mean, latent_standard_dev


class Decoder(nn.Module):
    """
    
    DEFINE A DECODER THAT UPSAMPLES

    """
    def __init__(self,
                 task,
                 hidden_size,
                 latent_size,
                 activation=nn.ReLU,
                 kernel_size=3,
                 num_blocks=4) -> None:
        super(Decoder, self).__init__()

        self.task = task
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.activation = activation
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks

        self.input_shape = task.input_shape
        self.shape_before_flat = [self.input_shape[0] // (2 ** (num_blocks - 1)), hidden_size]

    def forward(self, inputs):
        # the input layer of an nn module
        x = nn.Linear(self.latent_size, np.prod(self.shape_before_flat))(inputs)
        x = torch.reshape(x, [-1] + self.shape_before_flat)  # N * L * C 


        # add several residual blocks to the model
        for i in reversed(range(self.num_blocks)):

            if i > 0:
                # up-sample the sequence
                x = x.repeat([1, 2, 1])
                # torch has not symmetric padding
                x = F.pad(x, pad=[0, 0, 0, (self.input_shape[0] // (2 ** (i - 1))) % 2, 0, 0], mode='reflect')

            # the exponent of a positional embedding
            inverse_frequency = 1.0 / (10000.0 ** (torch.arange(0.0, self.hidden_size, 2.0) / self.hidden_size)).unsqueeze(dim=0)

            # calculate a positional embedding to break symmetry
            pos = torch.arange(0.0, x.shape[1], 1.0).unsqueeze(dim=-1)
            positional_embedding = torch.cat([torch.sin(pos * inverse_frequency), torch.cos(pos * inverse_frequency)], dim=1).unsqueeze(dim=0)

            # add the positional encoding
            h = torch.add(x, positional_embedding)  # N * L * C
            h = nn.LayerNorm(normalized_shape=h.shape[1:])(h)
            
            h = torch.transpose(h, 1, 2)  # N * C * L
            # first convolution layer in a residual block
            h = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, padding='same')(h)
            h = nn.LayerNorm(normalized_shape=h.shape[1:])(h)
            h = eval(self.activation)()(h)

            # second convolution layer in a residual block
            h = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, padding='same')(h)
            h = nn.LayerNorm(normalized_shape=h.shape[1:])(h)
            h = eval(self.activation)()(h)

            h = torch.transpose(h, 1, 2)  # N * L * C
            # add a residual connection to the model
            x = torch.add(x, h)
        
        # flatten the result and predict the params of a gaussian
        shaped_x = torch.reshape(x, [np.prod(x.shape[:-1]), x.shape[-1]])  # (N * L) * C
        logits = nn.Linear(x.shape[-1], self.task.num_classes)(shaped_x)
        logits = torch.reshape(logits, x.shape[:-1] + [self.task.num_classes])  # N * L * C

        return logits


class SequentialVAE(nn.Module):

    def __init__(self,
                 task,
                 hidden_size=64,
                 latent_size=256,
                 activation=nn.ReLU,
                 kernel_size=3,
                 num_blocks=4) -> None:

        super(SequentialVAE, self).__init__()

        self.encoder_cnn = Encoder(task, hidden_size, latent_size, activation, kernel_size, num_blocks)
        self.decoder_cnn = Decoder(task, hidden_size, latent_size, activation, kernel_size, num_blocks)
    
    def encode(self, x_batch):
        mean, standard_dev = self.encoder_cnn(x_batch)
        return torch.distributions.MultivariateNormal(loc=mean, scale_tril=torch.diag(standard_dev))

    def decode(self, z):
        logits = self.decoder_cnn(z)
        return torch.distributions.Categorical(logits=logits)

    def generate(self, z):
        logits = self.decoder_cnn(z)
        return torch.argmax(logits, dim=2)


class VAETrainer(nn.Module):
    
    def __init__(self,
                 vae: SequentialVAE,
                 optim=torch.optim.Adam,
                 lr=0.001,
                 beta=1.0) -> None:
        """Build a trainer for an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        vae: torch.nn.Module
            a list of keras model that predict distributions over scores
        optim: __class__
            the optimizer class to use for optimizing the oracle model
        lr: float
            the learning rate for the vae model optimizer
        """

        super(VAETrainer, self).__init__()
        self.vae = vae
        self.beta = beta

        # create optimizers for vae model
        self.vae_optim = optim(self.vae.parameters(), lr=lr)

    
    def train_step(self, x):
        statistics = dict()

        latent = self.vae.encode(x)
        z = latent.mean()
        prediction = self.vae.decode(z)

        nll = -prediction.log_prob(x)

        kld = torch.distributions.kl.kl_divergence(latent, torch.distributions.MultivariateNormal(loc=torch.zeros_like(z), scale_tril=torch.diag(torch.ones_like(z))))

        total_loss = torch.mean(nll) + torch.mean(kld) * self.beta

        self.vae_optim.zero_grad()
        total_loss.backward()
        self.vae_optim.step()

        statistics[f'vae/train/nll'] = nll
        statistics[f'vae/train/kld'] = kld

        return statistics
    

    def validate_step(self, x):
        statistics = dict()

        latent = self.vae.encode(x)
        z = latent.mean()
        prediction = self.vae.decode(z)

        nll = -prediction.log_prob(x)

        kld = torch.distributions.kl.kl_divergence(latent, torch.distributions.MultivariateNormal(loc=torch.zeros_like(z), scale_tril=torch.diag(torch.ones_like(z))))

        statistics[f'vae/validate/nll'] = nll
        statistics[f'vae/validate/kld'] = kld

        return statistics


    def _train(self, dataset):
        self.train()
        
        statistics = defaultdict(list)
        for train_step, (x, y) in enumerate(dataset):
            for name, tensor in self.train_step(x).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = torch.cat(statistics[name], dim=0)

        return statistics
    

    def _validate(self, dataset):
        self.eval()
        
        statistics = defaultdict(list)
        for train_step, (x, y) in enumerate(dataset):
            for name, tensor in self.train_step(x).items():
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

        train_data: tf.data.Dataset
            the training dataset already batched and prefetched
        validate_data: tf.data.Dataset
            the validation dataset already batched and prefetched
        logger: Logger
            an instance of the logger used for writing to tensor board
        epochs: int
            the number of epochs through the data sets to take
        """

        for e in range(epochs):
            logger.logger.info('Epoch [{}/{}]'.format(e, epochs-1))
            for name, loss in self._train(train_data).items():
                logger.record(name, loss, e)
            for name, loss in self._validate(validate_data).items():
                logger.record(name, loss, e)
    