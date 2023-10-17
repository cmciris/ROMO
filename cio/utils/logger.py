import torch
import numpy as np
import os
import sys
import pdb
import logging

from torch.utils.tensorboard import SummaryWriter


def get_logger(logging_dir, module_name, logging_filename='info.log', level=logging.INFO):
    """
    Get logger
    :return:
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(logging_dir, logging_filename))
    # Add console handler
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info('Log directory: %s', logging_dir)
    return logger

class Logger(object):
    def __init__(self, logging_dir, module_name, logging_filename) -> None:
        """
        Creates a logging interface to a tensorboard file for visualizing in the tensorboard web interface; note that mean, max, min, and std are recorded
        
        Arguments:
        
        logging_dir: str
            the path on the disk to save records to
        module_name: str
            __name__ of the module where instantiate Logger() class, __name__ is the module's name in the Python package namespace
        logging_filename: str
            the filename on the disk to save logs to
        """

        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        self.writer = SummaryWriter(logging_dir)
        self.logger = get_logger(logging_dir, module_name, logging_filename)

    def record(self, key, value, step, percentile=False):
        """
        Log statistics about training data to tensorboard log files for visualization later

        Arguments:

        key: str
            the string name to use when logging data in tensorboard that determines groupings in the web interface
        values: torch.tensor
            the tensor of values to record statistics about typically is multi dimensional
        step: int
            the total number of environment steps collected so far typically on intervals of 10000
        """

        step = torch.tensor(step, dtype=torch.int64)

        # may cause warning "NaN or Inf found in input tensor?"

        if torch.numel(value) == 1:
            
            # log one statistic of the incoming tensors
            self.writer.add_scalar(key, torch.reshape(value, []), global_step=step)

            message = '({}) value: {:.6f}'.format(key, float(value))
            self.logger.info(message)
        
        elif percentile:

            # log several statistics of the incoming tensors
            self.writer.add_scalar(key + '/100th', 
                                   torch.quantile(value, 1.0), 
                                   global_step=step)
            self.writer.add_scalar(key + '/90th',
                                   torch.quantile(value, 0.9),
                                   global_step=step)
            self.writer.add_scalar(key + '/80th',
                                   torch.quantile(value, 0.8),
                                   global_step=step)
            self.writer.add_scalar(key + '/50th',
                                   torch.quantile(value, 0.5),
                                   global_step=step)
            
            message = '({}) 100th: {:.6f}, 90th: {:.6f}, 80th: {:.6f}, 50th: {:.6f}'.format(key, torch.quantile(value, 1.0), torch.quantile(value, 0.9), torch.quantile(value, 0.8), torch.quantile(value, 0.5))
            self.logger.info(message)
        
        else:
            # log several statistics of the incoming tensors
            self.writer.add_scalar(key + '/max',
                                   torch.max(value),
                                   global_step=step)
            self.writer.add_scalar(key + '/mean',
                                   torch.mean(value),
                                   global_step=step)
            self.writer.add_scalar(key + '/min',
                                   torch.min(value),
                                   global_step=step)
            self.writer.add_scalar(key + '/std',
                                   torch.std(value),
                                   global_step=step)
            
            message = '({}) mean: {:.6f}, std: {:.6f}, max: {:.6f}, min: {:.6f}'.format(key, torch.mean(value), torch.std(value), torch.max(value), torch.min(value))
            self.logger.info(message)