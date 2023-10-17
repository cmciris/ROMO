from design_bench.task import Task
from design_bench import make

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pdb
from typing import Tuple, Any

class StaticGraphTask(Task):
    """A container class for model-based optimization problems where a
    dataset is paired with a ground truth score function such as a
    neural network learned with gradient descent, where the goal
    is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    is_discrete: bool
        An attribute that specifies whether the task dataset is discrete or
        continuous determined by whether the dataset instance task.dataset
        inherits from DiscreteDataset or ContinuousDataset

    dataset_name: str
        An attribute that specifies the name of a model-based optimization
        dataset, which might be used when labelling plots in a diagram of
        performance in a research paper using design-bench
    oracle_name: str
        Attribute that specifies the name of a machine learning model used
        as the ground truth score function for a model-based optimization
        problem; for example, "random_forest"
    x_name: str
        An attribute that specifies the name of designs in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper
    y_name: str
        An attribute that specifies the name of predictions in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper

    x: np.ndarray
        the design values 'x' for a model-based optimization problem
        represented as a numpy array of arbitrary type
    input_shape: Tuple[int]
        the shape of a single design values 'x', represented as a list of
        integers similar to calling np.ndarray.shape
    input_size: int
        the total number of components in the design values 'x', represented
        as a single integer, the product of its shape entries
    input_dtype: np.dtype
        the data type of the design values 'x', which is typically either
        floating point or integer (np.float32 or np.int32)

    y: np.ndarray
        the prediction values 'y' for a model-based optimization problem
        represented by a scalar floating point value per 'x'
    output_shape: Tuple[int]
        the shape of a single prediction value 'y', represented as a list of
        integers similar to calling np.ndarray.shape
    output_size: int
        the total number of components in the prediction values 'y',
        represented as a single integer, the product of its shape entries
    output_dtype: np.dtype
        the data type of the prediction values 'y', which is typically a
        type of floating point (np.float32 or np.float16)

    dataset_size: int
        the total number of paired design values 'x' and prediction values
        'y' in the dataset, represented as a single integer
    dataset_max_percentile: float
        the percentile between 0 and 100 of prediction values 'y' above
        which are hidden from access by members outside the class
    dataset_min_percentile: float
        the percentile between 0 and 100 of prediction values 'y' below
        which are hidden from access by members outside the class
    dataset_max_output: float
        the specific cutoff threshold for prediction values 'y' above
        which are hidden from access by members outside the class
    dataset_min_output: float
        the specific cutoff threshold for prediction values 'y' below
        which are hidden from access by members outside the class

    is_normalized_x: bool
        a boolean indicator that specifies whether the design values in
        the dataset are being normalized
    is_normalized_y: bool
        a boolean indicator that specifies whether the prediction values
        in the dataset are being normalized

    --- for discrete tasks only

    is_logits: bool (only supported for a DiscreteDataset)
        a value that indicates whether the design values contained in the
        model-based optimization dataset have already been converted to
        logits and need not be converted again
    num_classes: int
        an integer representing the number of classes in the distribution
        that the integer data points are sampled from which cannot be None
        and must also be greater than 1

    Public Methods:

    predict(np.ndarray) -> np.ndarray:
        a function that accepts a batch of design values 'x' as input and for
        each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

    dataset_to_oracle_x(np.ndarray) -> np.ndarray
        Helper function for converting from designs contained in the
        dataset format into a format the oracle is expecting to process,
        such as from integers to logits of a categorical distribution
    dataset_to_oracle_y(np.ndarray) -> np.ndarray
        Helper function for converting from predictions contained in the
        dataset format into a format the oracle is expecting to process,
        such as from normalized to denormalized predictions
    oracle_to_dataset_x(np.ndarray) -> np.ndarray
        Helper function for converting from designs in the format of the
        oracle into the design format the dataset contains, such as
        from categorical logits to integers
    oracle_to_dataset_y(np.ndarray) -> np.ndarray
        Helper function for converting from predictions in the
        format of the oracle into a format the dataset contains,
        such as from normalized to denormalized predictions

    iterate_batches(batch_size: int, return_x: bool,
                    return_y: bool, drop_remainder: bool)
                    -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model
    iterate_samples(return_x: bool, return_y: bool):
                    -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

    normalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance
    denormalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and undoes standardization so that they have their
        original empirical mean and variance
    normalize_y(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point prediction values 'y'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance
    denormalize_y(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point prediction values 'y'
        as input and undoes standardization so that they have their
        original empirical mean and variance

    map_normalize_x():
        a destructive function that standardizes the design values 'x'
        in the class dataset in-place so that they have zero empirical
        mean and unit variance
    map_denormalize_x():
        a destructive function that undoes standardization of the
        design values 'x' in the class dataset in-place which are expected
        to have zero  empirical mean and unit variance
    map_normalize_y():
        a destructive function that standardizes the prediction values 'y'
        in the class dataset in-place so that they have zero empirical
        mean and unit variance
    map_denormalize_y():
        a destructive function that undoes standardization of the
        prediction values 'y' in the class dataset in-place which are
        expected to have zero empirical mean and unit variance

    --- for discrete tasks only

    to_logits(np.ndarray) -> np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of integers as input and converts them to floating point
        logits of a certain probability distribution
    to_integers(np.ndarray) -> np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of floating point logits as input and converts them to integer
        representing the max of the distribution

    map_to_logits():
        a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits
    map_to_integers():
        a function that processes the dataset corresponding to this
        model-based optimization problem, and converts a floating point
        representation as logits to integers

    """

    def __init__(self,  task_name, **task_kwargs):
        """An interface to a static-graph task which includes a validation
        set and a non differentiable score function

        Args:

        task_name: str
            the name to a valid task using design_bench.make(task_name)
            such as 'HopperController-Exact-v0'
        **task_kwargs: dict
            additional keyword arguments that are passed to the design_bench task
            when it is created using design_bench.make

        """
        # handle relabel manually
        self.relabel = False
        if task_kwargs['relabel'] == True and 'Hopper' in task_name:
            self.relabel = True
            task_kwargs['relabel'] = False
        
        # use the design_bench registry to make a task
        self.wrapped_task = make(task_name, **task_kwargs)

        if self.relabel:
            self.y_relabel = self.run_relabel()
    
    def run_relabel(self, batch_size=256):
        y = []
        size = self.x.shape[0]
        start_ind = 0
        while start_ind + batch_size < size:
            end_ind = start_ind + batch_size
            print(start_ind, end_ind)
            x_batch = self.x[start_ind:end_ind]
            y_relabel = self.predict_numpy(x_batch)
            y.append(y_relabel)
            start_ind = end_ind
        x_batch = self.x[start_ind:]
        y_relabel = self.predict_numpy(x_batch)
        y.append(y_relabel)
        return np.concatenate(y, axis=0)

    @property
    def is_discrete(self):
        """Attribute that specifies whether the task dataset is discrete or
        continuous, determined by whether the dataset instance task.dataset
        inherits from DiscreteDataset or ContinuousDataset

        """

        return self.wrapped_task.is_discrete

    @property
    def oracle_name(self):
        """Attribute that specifies the name of a machine learning model used
        as the ground truth score function for a model-based optimization
        problem; for example, "random_forest"

        """

        return self.wrapped_task.oracle_name

    @property
    def dataset_name(self):
        """An attribute that specifies the name of a model-based optimization
        dataset, which might be used when labelling plots in a diagram of
        performance in a research paper using design-bench

        """

        return self.wrapped_task.dataset_name

    @property
    def x_name(self):
        """An attribute that specifies the name of designs in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper

        """

        return self.wrapped_task.x_name

    @property
    def y_name(self):
        """An attribute that specifies the name of predictions in a
        model-based optimization dataset, which might be used when labelling
        plots in a visualization of performance in a research paper

        """

        return self.wrapped_task.y_name

    @property
    def dataset_size(self):
        """the total number of paired design values 'x' and prediction values
        'y' in the dataset, represented as a single integer

        """

        return self.wrapped_task.dataset_size

    @property
    def dataset_max_percentile(self):
        """the percentile between 0 and 100 of prediction values 'y' above
        which are hidden from access by members outside the class

        """

        return self.wrapped_task.dataset_max_percentile

    @property
    def dataset_min_percentile(self):
        """the percentile between 0 and 100 of prediction values 'y' below
        which are hidden from access by members outside the class

        """

        return self.wrapped_task.dataset_min_percentile

    @property
    def dataset_max_output(self):
        """the specific cutoff threshold for prediction values 'y' above
        which are hidden from access by members outside the class

        """

        return self.wrapped_task.dataset_max_output

    @property
    def dataset_min_output(self):
        """the specific cutoff threshold for prediction values 'y' below
        which are hidden from access by members outside the class

        """

        return self.wrapped_task.dataset_min_output

    @property
    def input_shape(self):
        """the shape of a single design values 'x', represented as a list of
        integers similar to calling np.ndarray.shape

        """

        return self.wrapped_task.input_shape

    @property
    def input_size(self):
        """the total number of components in the design values 'x',
        represented as a single integer, the product of its shape entries

        """

        return self.wrapped_task.input_size

    @property
    def input_dtype(self):
        """the data type of the design values 'x', which is typically either
        floating point or integer (np.float32 or np.int32)

        """

        return self.wrapped_task.input_dtype

    @property
    def output_shape(self):
        """the shape of a single prediction value 'y', represented as a list
        of integers similar to calling np.ndarray.shape

        """

        return self.wrapped_task.output_shape

    @property
    def output_size(self):
        """the total number of components in the prediction values 'y',
        represented as a single integer, the product of its shape entries

        """

        return self.wrapped_task.output_size

    @property
    def output_dtype(self):
        """the data type of the prediction values 'y', which is typically a
        type of floating point (np.float32 or np.float16)

        """

        return self.wrapped_task.output_dtype

    @property
    def x(self):
        """the design values 'x' for a model-based optimization problem
        represented as a numpy array of arbitrary type

        """

        return self.wrapped_task.x

    @property
    def y(self):
        """the prediction values 'y' for a model-based optimization problem
        represented by a scalar floating point value per 'x'

        """

        return self.wrapped_task.y

    @property
    def is_normalized_x(self):
        """a boolean indicator that specifies whether the design values in
        the dataset are being normalized

        """

        return self.wrapped_task.is_normalized_x

    @property
    def is_normalized_y(self):
        """a boolean indicator that specifies whether the prediction values
        in the dataset are being normalized

        """

        return self.wrapped_task.is_normalized_y

    @property
    def is_logits(self):
        """a value that indicates whether the design values contained in the
        model-based optimization dataset have already been converted to
        logits and need not be converted again

        """

        return self.wrapped_task.is_logits

    @property
    def num_classes(self):
        """an integer representing the number of classes in the distribution
        that the integer data points are sampled from which cannot be None
        and must also be greater than 1

        """

        return self.wrapped_task.num_classes

    def iterate_batches(self, batch_size, return_x=True,
                        return_y=True, drop_remainder=False):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Arguments:

        batch_size: int
            a positive integer that specifies the batch size of samples
            taken from a model-based optimization data set; batches
            with batch_size elements are yielded
        return_x: bool
            a boolean indicator that specifies whether the generator yields
            design values at every iteration; note that at least one of
            return_x and return_y must be set to True
        return_y: bool
            a boolean indicator that specifies whether the generator yields
            prediction values at every iteration; note that at least one
            of return_x and return_y must be set to True
        drop_remainder: bool
            a boolean indicator representing whether the last batch
            should be dropped in the case it has fewer than batch_size
            elements; the default behavior is not to drop the smaller batch.

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        return iter(self.wrapped_task.iterate_batches(
            batch_size, return_x=return_x,
            return_y=return_y, drop_remainder=drop_remainder))

    def iterate_samples(self, return_x=True, return_y=True):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Arguments:

        return_x: bool
            a boolean indicator that specifies whether the generator yields
            design values at every iteration; note that at least one of
            return_x and return_y must be set to True
        return_y: bool
            a boolean indicator that specifies whether the generator yields
            prediction values at every iteration; note that at least one
            of return_x and return_y must be set to True

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        return iter(self.wrapped_task.iterate_samples(return_x=return_x,
                                                      return_y=return_y))

    def __iter__(self):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        return iter(self.wrapped_task)

    def map_normalize_x_numpy(self):
        """a function that standardizes the design values 'x' to have zero
        empirical mean and unit empirical variance in the dataset

        """

        self.wrapped_task.map_normalize_x()

    # @tf.function
    def map_normalize_x(self):
        """a function that standardizes the design values 'x' to have zero
        empirical mean and unit empirical variance in the dataset

        """

        self.map_normalize_x_numpy()
        
    def map_normalize_y_numpy(self):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance in the dataset

        """

        self.wrapped_task.map_normalize_y()

        if self.relabel:
            self.y_relabel = self.normalize_y_numpy(self.y_relabel)

    # @tf.function
    def map_normalize_y(self):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance in the dataset

        """

        self.map_normalize_y_numpy()

    def map_denormalize_x_numpy(self):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance in the dataset

        """

        self.wrapped_task.map_denormalize_x()

    # @tf.function
    def map_denormalize_x(self):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance in the dataset

        """

        self.map_denormalize_x_numpy()

    def map_denormalize_y_numpy(self):
        """a function that un-standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance in the dataset

        """

        self.wrapped_task.map_denormalize_y()

    # @tf.function
    def map_denormalize_y(self):
        """a function that un-standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance in the dataset

        """

        self.map_denormalize_y_numpy()

    def map_to_integers_numpy(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts a floating point
        representation as logits to integers

        """

        self.wrapped_task.map_to_integers()

    # @tf.function
    def map_to_integers(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts a floating point
        representation as logits to integers

        """

        self.map_to_integers_numpy()

    def map_to_logits_numpy(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits

        """

        self.wrapped_task.map_to_logits()

    # @tf.function
    def map_to_logits(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits

        """

        self.map_to_logits_numpy()

    def normalize_x_numpy(self, x):
        """a function that standardizes the design values 'x' to have
        zero empirical mean and unit empirical variance

        Arguments:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            shall be normalized according to dataset statistics

        Returns:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            has been normalized using dataset statistics

        """

        return self.wrapped_task.normalize_x(x)

    # @tf.function
    def normalize_x(self, x):
        """a function that standardizes the design values 'x' to have
        zero empirical mean and unit empirical variance

        Arguments:

        x: torch.Tensor
            a design value represented as a numpy array potentially
            given as a batch of designs which
            shall be normalized according to dataset statistics

        Returns:

        x: torch.Tensor
            a design value represented as a numpy array potentially
            given as a batch of designs which
            has been normalized using dataset statistics

        """

        device = x.device
        dtype = x.dtype
        x = x.detach().cpu().numpy()
        new_x = torch.tensor(self.normalize_x_numpy(x), device=device, dtype=dtype)
        new_x = new_x.reshape(x.shape)
        return new_x

    def normalize_y_numpy(self, y):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance

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

        return self.wrapped_task.normalize_y(y)

    # @tf.function
    def normalize_y(self, y):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance

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

        device = y.device
        dtype = y.dtype
        y = y.detach().cpu().numpy()
        new_y = torch.tensor(self.normalize_y_numpy(y), device=device, dtype=dtype)
        new_y = new_y.reshape(y.shape)
        return new_y

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

        return self.wrapped_task.denormalize_x(x)

    # @tf.function
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

        device = x.device
        dtype = x.dtype
        x = x.detach().cpu().numpy()
        new_x = torch.tensor(self.denormalize_x_numpy(x), device=device, dtype=dtype)
        new_x = new_x.reshape(x.shape)
        return new_x

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

        return self.wrapped_task.denormalize_y(y)

    # @tf.function
    def denormalize_y(self, y):
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

        device = y.device
        dtype = y.dtype
        y = y.detach().cpu().numpy()
        new_y = torch.tensor(self.denormalize_y_numpy(y), device=device, dtype=dtype)
        new_y = new_y.reshape(y.shape)
        return new_y

    def to_integers_numpy(self, x):
        """A helper function that accepts design values represented as a numpy
        array of floating point logits as input and converts them to integer
        representing the max of the distribution

        Arguments:

        x: np.ndarray
            a numpy array containing design values represented as floating
            point numbers which have be converted from integer samples of
            a certain probability distribution

        Returns:

        x: np.ndarray
            a numpy array containing design values represented as integers
            which have been converted from a floating point
            representation of a certain probability distribution

        """

        return self.wrapped_task.to_integers(x)

    # @tf.function
    def to_integers(self, x):
        """A helper function that accepts design values represented as a numpy
        array of floating point logits as input and converts them to integer
        representing the max of the distribution

        Arguments:

        x: torch.Tensor
            a numpy array containing design values represented as floating
            point numbers which have be converted from integer samples of
            a certain probability distribution

        Returns:

        x: torch.Tensor
            a numpy array containing design values represented as integers
            which have been converted from a floating point
            representation of a certain probability distribution

        """

        device = x.device
        dtype = x.dtype
        x = x.detach().cpu().numpy()
        new_x = torch.tensor(self.to_integers_numpy(), device=device, dtype=dtype)
        new_x = new_x.reshape(x.shape[:-1])
        return new_x

    def to_logits_numpy(self, x):
        """A helper function that accepts design values represented as a numpy
        array of integers as input and converts them to floating point
        logits of a certain probability distribution

        Arguments:

        x: np.ndarray
            a numpy array containing design values represented as integers
            which are going to be converted into a floating point
            representation of a certain probability distribution

        Returns:

        x: np.ndarray
            a numpy array containing design values represented as floating
            point numbers which have be converted from integer samples of
            a certain probability distribution

        """

        return self.wrapped_task.to_logits(x)

    # @tf.function
    def to_logits(self, x):
        """A helper function that accepts design values represented as a numpy
        array of integers as input and converts them to floating point
        logits of a certain probability distribution

        Arguments:

        x: torch.Tensor
            a numpy array containing design values represented as integers
            which are going to be converted into a floating point
            representation of a certain probability distribution

        Returns:

        x: torch.Tensor
            a numpy array containing design values represented as floating
            point numbers which have be converted from integer samples of
            a certain probability distribution

        """

        device = x.device
        dtype = x.dtype
        x = x.detach().cpu().numpy()
        new_x = torch.tensor(self.to_logits_numpy(x), device=device, dtype=dtype)
        new_x = new_x.reshape(list(x.shape) + [self.wrapped_task.dataset.num_classes - 1])
        return new_x

    def predict_numpy(self, x_batch):
        """a function that accepts a batch of design values 'x' as input and
        for each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        return self.wrapped_task.predict(x_batch)

    # @tf.function
    def predict(self, x):
        """a function that accepts a batch of design values 'x' as input and
        for each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

        Arguments:

        x_batch: torch.Tensor
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y_batch: torch.Tensor
            a batch of prediction values 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        device = x.device
        dtype = x.dtype
        x = x.detach().cpu().numpy()
        y = torch.tensor(self.predict_numpy(x), device=device, dtype=dtype)
        y = y.reshape([x.shape[0], 1])
        return y

    
class MBODataSet(Dataset):
    def __init__(self, tuples) -> None:
        self.datas = tuples[0]
        self.labels = tuples[1]

        assert len(self.datas) == len(self.labels)
        self._len = len(self.datas)
    
    def __getitem__(self, index) -> Any:
        data = self.datas[index]
        label = self.labels[index]
        return data, label
    
    def __len__(self):
        return self._len
    

def build_pipeline(x, y, w=None, val_size=200, batch_size=128, bootstraps=0, bootstraps_noise=None, buffer=None):
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
    train_inputs = [x[val_size:], y[val_size:]]
    validate_inputs = [x[:val_size], y[:val_size]]
    size = x.shape[0] - val_size

    if bootstraps > 0:

        # sample the data set with replacement
        bootstraps_indices = torch.randint(low=0, high=size, size=[bootstraps, size], dtype=torch.long)
        bootstraps_train_inputs = [data[bootstraps_indices] for data in train_inputs]

        # add noise to the labels to increase diversity
        if bootstraps_noise is not None:
            bootstraps_train_inputs[1] = bootstraps_train_inputs[1] + bootstraps_noise * torch.randn_like(bootstraps_train_inputs[1])
            bootstraps_train_inputs[2] = bootstraps_train_inputs[2] + bootstraps_noise * torch.randn_like(bootstraps_train_inputs[2])
    
    # build the parallel torch data loading pipeline
    if bootstraps > 0:
        bootstraps_training_dataset = [MBODataSet([bootstraps_train_inputs[0][b], bootstraps_train_inputs[1][b], bootstraps_train_inputs[2][b]]) for b in range(bootstraps)]
        validation_dataset = MBODataSet(validate_inputs)
        training_dataloader = [DataLoader(_training_dataset, batch_size=batch_size, shuffle=True) for _training_dataset in bootstraps_training_dataset]
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    else:
        training_dataset = MBODataSet(train_inputs)
        validation_dataset = MBODataSet(validate_inputs)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    # batch and prefetch each data set
    return (training_dataloader, validation_dataloader)


def build_rim_pipeline(x, y, val_size=200, train_ratio=0.5, batch_size=128, bootstraps=0, bootstraps_noise=None, buffer=None):
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
    data_size = x.shape[0]
    train_size = int(train_ratio * (data_size - val_size))
    pool_size = data_size - val_size - train_size
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
        bootstraps_training_dataset = [MBODataSet([bootstraps_train_inputs[0][b], bootstraps_train_inputs[1][b]]) for b in range(bootstraps)]
        validation_dataset = MBODataSet(validate_inputs)
        training_dataloader = [DataLoader(_training_dataset, batch_size=batch_size, shuffle=True) for _training_dataset in bootstraps_training_dataset]
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    else:
        training_dataset = MBODataSet(train_inputs)
        validation_dataset = MBODataSet(validate_inputs)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    # batch and prefetch each data set
    return (training_dataloader, validation_dataloader, pool_inputs)
