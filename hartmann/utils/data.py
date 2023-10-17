import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pdb
from typing import Tuple, Any

class HartmannTask(object):
    def __init__(self, data_dir, ** task_kwargs) -> None:
        """
        An interface to a branin task which includes a validation set and a differentiable score functionn
        Args:
            data_dir: str
            **task_kwargs: dict
                additional keyword arguments that are passed to the task when it is created
        """
        dataset_kwargs = task_kwargs.get('dataset_kwargs', dict({}))
        self.max_samples = dataset_kwargs.get('max_samples')
        self.distribution = dataset_kwargs.get('distribution')
        self.x_opt_channel = dataset_kwargs.get('opt_channel')
        self.x_opt_ub = dataset_kwargs.get('opt_ub')
        self.x_opt_lb = dataset_kwargs.get('opt_lb')

        if task_kwargs.get('relabel', False):
            self.x = np.load(os.path.join(data_dir, 'relabel/x.npy')).astype(np.float32)
            self.y = np.load(os.path.join(data_dir, 'relabel/y.npy')).astype(np.float32)
        else:
            self.x = np.load(os.path.join(data_dir, 'x.npy')).astype(np.float32)
            self.y = np.load(os.path.join(data_dir, 'y.npy')).astype(np.float32)

        if self.max_samples is not None:
            self.x = self.x[:self.max_samples]
            self.y = self.y[:self.max_samples]
        self.is_discrete = False  # for branin

        self.is_score_scale = task_kwargs.get('score_scale', False)
        self.score_min = None
        self.score_max = None
        self.score_range = None
        if self.is_score_scale:
            self.score_range = eval(task_kwargs.get('score_range', '(0, 100)'))
            self.min_max_score_scale(self.score_range)
        
        self.x, self.y = self.clip_outliers(self.x, self.y, k=1000)

        self.is_normalize_y = False
        self.mu_y = None
        self.st_y = None
        self.is_normalize_aux = False
        self.mu_aux = None
        self.st_aux = None
        self.is_normalize_x = False
        self.mu_x = None
        self.st_x = None

    @staticmethod
    def clip_outliers(x, y, k=10) -> Tuple[np.ndarray, np.ndarray]:
        """
        clip outliers on cpu, delete max k samples & min k samples.
        """
        assert x.shape[0] == y.shape[0]
        _y = torch.tensor(y, dtype=torch.float32)
        indices_min = torch.topk(-_y[:, 0], k=k)[1]
        indices_max = torch.topk(_y[:, 0], k=k)[1]
        indices_clip = torch.cat([indices_min, indices_max], dim=0).numpy()
        x = np.delete(x, indices_clip, axis=0)
        y = np.delete(y, indices_clip, axis=0)
        return x, y
        
    def get_bound(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Get the optimization bound of task, None for no limit
        Returns:
            x_opt_channel: int
                operate optimization on the first 'x_opt_channel' dims of design space
            x_opt_ub: float
                upper bound of design space
            x_opt_lb: float
                lower bound of design space
        """
        if (self.x_opt_ub is not None) and (self.x_opt_lb is not None):
            assert np.all(self.x_opt_lb <= self.x_opt_ub)
        return self.x_opt_channel, self.x_opt_ub, self.x_opt_lb

    def min_max_score_scale(self, range: Tuple) -> None:
        """
        Scale score data 'y' to [range_low, range_high] interval
        Args:
            data: np.ndarray
                a batch of data inputs shaped like [batch_size, channels]
        Returns:
            data_scaled: np.ndarray
                a batch of data outputs shaped like [batch_size, 1]
        """
        assert len(range) == 2 and range[0] <= range[1]
        self.score_min = np.min(self.y)
        self.score_max = np.max(self.y)
        range_low = range[0]
        range_interval = range[1] - range[0]
        y_scaled = ((self.y - self.score_min) * 1.) / (self.score_max - self.score_min) * range_interval + range_low
        self.y = y_scaled
    
    def map_normalize_y(self):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance in the dataset
        call after min_max_score_scale!!!
        """
        # compute normalization statistics for the score
        if ~self.is_normalize_y:
            self.mu_y = np.mean(self.y, axis=0, keepdims=True).astype(np.float32)
            self.y = self.y - self.mu_y
            self.st_y = np.std(self.y, axis=0, keepdims=True).astype(np.float32).clip(1e-6, 1e9)
            self.y = self.y / self.st_y
            self.is_normalize_y = True
    
    def map_normalize_x(self):
        """a function that standardizes the design values 'x' to have zero
        empirical mean and unit empirical variance in the dataset
        """
        if ~self.is_normalize_x:
            self.mu_x = np.mean(self.x, axis=0, keepdims=True).astype(np.float32)
            self.x = self.x - self.mu_x
            self.st_x = np.std(self.x, axis=0, keepdims=True).astype(np.float32).clip(1e-6, 1e9)
            self.x = self.x / self.st_x
            if (self.x_opt_ub is not None) and (self.x_opt_lb is not None):
                self.x_opt_ub = (self.x_opt_ub - self.mu_x) / self.st_x
                self.x_opt_lb = (self.x_opt_lb - self.mu_x) / self.st_x
            self.is_normalize_x = True
    
    def map_to_logits(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits
        """
        pass

    def denormalize_x_numpy(self, x):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance
        Arguments:
            x: np.ndarray
                a design value represented as a numpy array potentially
                given as a batch of designs which
                shall be denormalized according to dataset statistics
        Returns:
            x: np.ndarray
                a design value represented as a numpy array potentially
                given as a batch of designs which
                has been denormalized using dataset statistics
        """
        if self.is_normalize_x:
            x = x * self.st_x + self.mu_x
        return x

    def denormalize_x(self, x):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance
        Arguments:
            x: torch.Tensor
                a design value represented as a numpy array potentially
                given as a batch of designs which
                shall be denormalized according to dataset statistics
        Returns:
            x: torch.Tensor
                a design value represented as a numpy array potentially
                given as a batch of designs which
                has been denormalized using dataset statistics
        """
        if self.is_normalize_x:
            st_x = torch.tensor(self.st_x, device=x.device)
            mu_x = torch.tensor(self.mu_x, device=x.device)
            x = x * st_x + mu_x
        return x
    
    def denormalize_y_numpy(self, y):
        """a function that un-standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance
        Arguments:
            y: np.ndarray
                a prediction value represented as a numpy array potentially
                given as a batch of predictions which
                shall be denormalized according to dataset statistics
        Returns:
            y: np.ndarray
                a prediction value represented as a numpy array potentially
                given as a batch of predictions which
                has been denormalized using dataset statistics
        """
        if self.is_normalize_y:
            y = y * self.st_y + self.mu_y
        return y

    def denormalize_y(self, y):
        """a function that un-standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance
        Arguments:
            y: torch.Tensor
                a prediction value represented as a numpy array potentially
                given as a batch of predictions which
                shall be denormalized according to dataset statistics
        Returns:
            y: torch.Tensor
                a prediction value represented as a numpy array potentially
                given as a batch of predictions which
                has been denormalized using dataset statistics
        """
        if self.is_normalize_y:
            st_y = torch.tensor(self.st_y, device=y.device)
            mu_y = torch.tensor(self.mu_y, device=y.device)
            y = y * st_y + mu_y
        return y
    
    def normalize_y_numpy(self, y):
        """a function that standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance
        Arguments:
            y: np.ndarray
                a prediction value represented as a numpy array potentially
                given as a batch of predictions which
                shall be normalized according to dataset statistics
        Returns:
            y: np.ndarray
                a prediction value represented as a numpy array potentially
                given as a batch of predictions which
                has been normalized using dataset statistics
        """
        if self.is_normalize_y:
            y = (y - self.mu_y) / self.st_y
        return y

    def normalize_y(self, y):
        """a function that standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance
        Arguments:
            y: torch.Tensor
                a prediction value represented as a numpy array potentially
                given as a batch of predictions which
                shall be normalized according to dataset statistics
        Returns:
            y: torch.Tensor
                a prediction value represented as a numpy array potentially
                given as a batch of predictions which
                has been normalized using dataset statistics
        """
        if self.is_normalize_y:
            st_y = torch.tensor(self.st_y, device=y.device)
            mu_y = torch.tensor(self.mu_y, device=y.device)
            y = (y - mu_y) / st_y
        return y

    
class HartmannDataSet(Dataset):
    def __init__(self, tuples) -> None:
        self.x = tuples[0]
        self.y = tuples[1]

        assert len(self.x) == len(self.y)
        self._len = len(self.x)
    
    def __getitem__(self, index) -> Any:
        x = self.x[index]
        y = self.y[index]
        return x, y
    
    def __len__(self):
        return self._len


def build_pipeline(x, y, val_size=200, train_size=800, batch_size=128, bootstraps=0, bootstraps_noise=None, buffer=None):
    """Split a model-based optimization dataset consisting of a set of design
    values x and prediction values y into a training and validation set,
    supporting bootstrapping and importance weighting
    Args:
        x: torch.Tensor
            a tensor containing design values from a model-based optimization
            dataset, typically taken from task.x
        y: torch.Tensor
            a tensor containing prediction values from a model-based optimization
            dataset, typically taken from task.y
        val_size: int
            the number of samples randomly chosen to be in the validation set
            returned by the function
        batch_size: int
            the number of samples to load in every batch when drawing samples
            from the training and validation sets
        bootstraps: int
            the number of copies of the dataset to draw with replacement
            for training an ensemble of forward models
        bootstraps_noise: float
            the standard deviation of zero mean gaussian noise independently
            sampled and added to each bootstrap of the dataset
    Returns:
        training_dataset: torch.utils.data.DataLoader
            a tensorflow dataset that has been batched and prefetched
            with an optional importance weight and optional bootstrap included
        validation_dataset: torch.utils.data.DataLoader
            a tensorflow dataset that has been batched and prefetched
            with an optional importance weight and optional bootstrap included
    """

    # shuffle the dataset using a common set of indices
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    # create a training and validation split
    x = x[indices]
    y = y[indices]
    train_end = val_size + train_size if val_size + train_size < len(x) else None
    train_inputs = [x[val_size:train_end], y[val_size:train_end]]
    validate_inputs = [x[:val_size], y[:val_size]]
    size = train_size if val_size + train_size < len(x) else x.shape[0] - val_size

    if bootstraps > 0:
        # sample the data set with replacement
        bootstraps_indices = torch.randint(low=0, high=size, size=[bootstraps, size], dtype=torch.long)
        bootstraps_train_inputs = [data[bootstraps_indices] for data in train_inputs]
        # add noise to the labels to increase diversity
        if bootstraps_noise is not None:
            bootstraps_train_inputs[1] = bootstraps_train_inputs[1] + bootstraps_noise * torch.randn_like(bootstraps_train_inputs[1])
    
    # build the parallel torch data loading pipeline
    if bootstraps > 0:
        bootstraps_training_dataset = [HartmannDataSet([bootstraps_train_inputs[0][b], bootstraps_train_inputs[1][b]]) for b in range(bootstraps)]
        validation_dataset = HartmannDataSet(validate_inputs)
        training_dataloader = [DataLoader(_training_dataset, batch_size=batch_size, shuffle=True) for _training_dataset in bootstraps_training_dataset]
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    else:
        training_dataset = HartmannDataSet(train_inputs)
        validation_dataset = HartmannDataSet(validate_inputs)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    # batch and prefetch each data set
    return (training_dataloader, validation_dataloader)


def build_rim_pipeline(x, y, pool_size=800, val_size=200, train_size=800, batch_size=128, bootstraps=0, bootstraps_noise=None, buffer=None):
    """Split a model-based optimization dataset consisting of a set of design
    values x and prediction values y into a training and validation set,
    supporting bootstrapping and importance weighting
    Args:
        x: torch.Tensor
            a tensor containing design values from a model-based optimization
            dataset, typically taken from task.x
        y: torch.Tensor
            a tensor containing prediction values from a model-based optimization
            dataset, typically taken from task.y
        val_size: int
            the number of samples randomly chosen to be in the validation set
            returned by the function
        batch_size: int
            the number of samples to load in every batch when drawing samples
            from the training and validation sets
        bootstraps: int
            the number of copies of the dataset to draw with replacement
            for training an ensemble of forward models
        bootstraps_noise: float
            the standard deviation of zero mean gaussian noise independently
            sampled and added to each bootstrap of the dataset
    Returns:
        training_dataset: torch.utils.data.DataLoader
            a tensorflow dataset that has been batched and prefetched
            with an optional importance weight and optional bootstrap included
        validation_dataset: torch.utils.data.DataLoader
            a tensorflow dataset that has been batched and prefetched
            with an optional importance weight and optional bootstrap included
    """

    # shuffle the dataset using a common set of indices
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    # create a training and validation split
    x = x[indices]
    y = y[indices]
    train_end = val_size + pool_size + train_size if val_size + pool_size + train_size < len(x) else None
    train_inputs = [x[val_size + pool_size : train_end], y[val_size + pool_size : train_end]]
    pool_inputs = [x[val_size : val_size + pool_size], y[val_size : val_size + pool_size]]
    validate_inputs = [x[:val_size], y[:val_size]]
    size = train_size if val_size + pool_size + train_size < len(x) else x.shape[0] - val_size - pool_size

    if bootstraps > 0:
        # sample the data set with replacement
        bootstraps_indices = torch.randint(low=0, high=size, size=[bootstraps, size], dtype=torch.long)
        bootstraps_train_inputs = [data[bootstraps_indices] for data in train_inputs]
        # add noise to the labels to increase diversity
        if bootstraps_noise is not None:
            bootstraps_train_inputs[1] = bootstraps_train_inputs[1] + bootstraps_noise * torch.randn_like(bootstraps_train_inputs[1])
    
    # build the parallel torch data loading pipeline
    if bootstraps > 0:
        bootstraps_training_dataset = [HartmannDataSet([bootstraps_train_inputs[0][b], bootstraps_train_inputs[1][b]]) for b in range(bootstraps)]
        validation_dataset = HartmannDataSet(validate_inputs)
        training_dataloader = [DataLoader(_training_dataset, batch_size=batch_size, shuffle=True) for _training_dataset in bootstraps_training_dataset]
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    else:
        training_dataset = HartmannDataSet(train_inputs)
        validation_dataset = HartmannDataSet(validate_inputs)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    # batch and prefetch each data set
    return (training_dataloader, validation_dataloader, pool_inputs)
