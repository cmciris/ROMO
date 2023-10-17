from models.forward_model import build_forward_model, build_oracle
from models.coms_model import ConservativeObjectiveModel
from models.oracle_model import HartmannOracle
from models.genetic import UniformCrossover, UniformMutate, NormalMutate
import torch
from scipy.stats import spearmanr
import numpy as np
import os
import copy
import json
import time
from utils.logger import Logger
from utils.data import HartmannTask, build_pipeline

import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_log_name(**kwargs):
    """
    get target log_dir from key word arguments and generate log directory
    :param kwargs: config key word arguments
    :return logging_filenmae: logging filename
    :return logging_dd
    """
    logging_dir = kwargs.get('logging_dir')
    exp = kwargs.get('exp')
    max_generations = kwargs.get('max_generations')
    evaluation_samples = kwargs.get('evaluation_samples')
    run_id = '%s' % (time.strftime('%m%d%H%M%S'))
    run_prefix = '%s_max_generations_%d_samples_%d' % (
        exp,
        max_generations,
        evaluation_samples[0] * evaluation_samples[1]
    )
    logging_filename = '%s_info.log' % run_id
    return logging_filename, run_id


def genetic(**config):
    logging_dir=config.get('logging_dir')
    data_dir=config.get('data_dir')
    model_dir=config.get('model_dir')
    task_score_scale=config.get('task_score_scale')
    task_score_range=config.get('task_score_range')
    task_relabel=config.get('task_relabel')
    task_max_samples=config.get('task_max_samples')
    task_distribution=config.get('task_distribution')
    task_opt_channel=config.get('task_opt_channel')
    task_opt_ub=config.get('task_opt_ub')
    task_opt_lb=config.get('task_opt_lb')
    normalize_ys=config.get('normalize_ys')
    normalize_xs=config.get('normalize_xs')
    in_latent_space=config.get('in_latent_space')
    max_generations=config.get('max_generations')
    evaluation_samples=config.get('evaluation_samples')
    fast=config.get('fast')
    elite_num = config.get('elite_num')
    crossover_rate = config.get('crossover_rate')
    mutation_rate = config.get('mutation_rate')
    
    # create the logger and export the experiment parameters
    logging_filename, run_id = _get_log_name(**config)
    logger = Logger(logging_dir, __name__, logging_filename)
    with open(os.path.join(logging_dir, "%s_params.json" % run_id), "w") as f:
        json.dump(config, f, indent=4)
    
    # create a model-based optimization task
    task = HartmannTask(data_dir,
                      score_scale=task_score_scale,
                      score_range=task_score_range,relabel=task_relabel,
                      dataset_kwargs=dict(max_samples=task_max_samples, distribution=task_distribution,
                      opt_channel=task_opt_channel,
                      opt_ub=task_opt_ub,
                      opt_lb=task_opt_lb))

    if normalize_ys:
        task.map_normalize_y()
    if task.is_discrete and not in_latent_space:
        task.map_to_logits()
    if normalize_xs:
        task.map_normalize_x()

    # save the initial dataset statistics for safe keeping
    x = torch.tensor(task.x, dtype=torch.float32, device=device)
    y = torch.tensor(task.y, dtype=torch.float32, device=device)
    
    # get the optimization limitation, i.e. bound of design space after normalization
    x_opt_channel, x_opt_ub, x_opt_lb = task.get_bound()
    if (x_opt_ub is not None) and (x_opt_lb is not None):
        x_opt_ub = torch.tensor(x_opt_ub, dtype=torch.float32, device=device)
        x_opt_lb = torch.tensor(x_opt_lb, dtype=torch.float32, device=device)
    opt_limit = dict(x_opt_channel=x_opt_channel,
                     x_opt_ub=x_opt_ub,
                     x_opt_lb=x_opt_lb)

    if task.is_discrete and in_latent_space:
        pass
        # vae
    logger.logger.info("Task created: {}".format(data_dir))
    
    input_shape = x.shape[1:]
    output_shape = y.shape[1:]

    logger.logger.info('Epoch [0/0]')
    logger.logger.info('(train/loss) mean: 0.0, std: 0.0, max: 0.0, min: 0.0')
    logger.logger.info('(validate/loss) value: 0.0')
    
    # -----------------------------------------------------------
    logger.logger.info(f"========== Offline Evaluation ==========")
    # pdb.set_trace()
    logger.logger.info("Evaluating {}".format('quickly and only log once!' if fast else 'now!'))
    # select the worst k initial designs from the dataset
    # indices = torch.topk(-y[:, 0], k=evaluation_samples)[1]
    indices = torch.zeros(0, device=device, dtype=int)
    bins = torch.linspace(x[:,-1].min(), x[:,-1].max(), steps=evaluation_samples[0])
    y_range = y.max() - y.min()
    for idx in range(len(bins) - 1):
        tmp = copy.deepcopy(y)
        tmp[(x[:,-1] > bins[idx]) & (x[:,-1] < bins[idx + 1])] -= y_range
        index = torch.topk(-tmp[:, 0], k=evaluation_samples[1])[1]
        indices = torch.concat([indices, index], dim=0)
    initial_x = torch.index_select(x, index=indices, dim=0)
    initial_y = torch.index_select(y, index=indices, dim=0)
    xt = initial_x
    evaluation_samples = initial_x.shape[0]

    logger.logger.info('Step [{}/{}]'.format(0, 1 + max_generations))
    if not fast:

        scores = []
        predictions = []
        solutions = []

        solution = xt
        if task.is_discrete and in_latent_space:
            pass
        
        with torch.no_grad():
            y_truth = HartmannOracle(solution)
        
        if normalize_ys:
            initial_y = task.denormalize_y(initial_y)

        if task.is_normalize_x:
            solution_to_save = task.denormalize_x(solution)
        else:
            solution_to_save = solution
        solutions.append(solution_to_save)  
        
        logger.record(f"dataset_score", initial_y, 0, percentile=True)
        logger.record(f"dataset_score", initial_y, 0)
        logger.record(f"score", y_truth, 0, percentile=True)
        logger.record(f"score", y_truth, 0)
    
    for step in range(1, 1 + max_generations):
        logger.logger.info('Step [{}/{}]'.format(step, max_generations))
        
        # update the set of solution particles
        with torch.no_grad():
            y_truth = HartmannOracle(solution)

        _, index = torch.sort(y_truth, dim=0, descending=True)
        elimated_index = index.reshape(-1)[elite_num:]

        nxt_generation = []
        nc = int((evaluation_samples - elite_num) / 2)
        for i in range(nc):
            parents = torch.randint(low=0, high=evaluation_samples-1, size=(2,))
            parent1, parent2 = xt[parents[0], 0:x_opt_channel], xt[parents[1], 0:x_opt_channel]
            child1, child2 = UniformCrossover(parent1, parent2, crossover_rate)
            nxt_generation.append(child1)
            nxt_generation.append(child2)
        nxt_generation = torch.stack(nxt_generation)
        
        for i in range(nc * 2):
            nxt_generation[i] = NormalMutate(nxt_generation[i], mutation_rate, opt_limit)
        
        xt[elimated_index, 0:x_opt_channel] = nxt_generation
        
        if not fast or step == max_generations:

            solution = xt
            if task.is_discrete and in_latent_space:
                pass
                
            if task.is_normalize_x:
                solution_to_save = task.denormalize_x(solution)
            else:
                solution_to_save = solution

            np.save(os.path.join(logging_dir, "{}_solution.npy".format(run_id)), solution_to_save.detach().cpu().numpy())
                
            # evaluate the solutions found by the model
            y_truth = HartmannOracle(solution)
            prediction = y_truth

            # record the prediction and score to the logger
            logger.record(f"score", y_truth, step, percentile=True)
            logger.record(f"score", y_truth, step)
            logger.record(f"solver/distance", torch.linalg.norm(xt - initial_x), step)
            logger.record(f"solver/prediction", prediction, step, percentile=True)
            logger.record(f"solver/prediction", prediction, step)
        
        if not fast:
            
            solutions.append(solution_to_save)
            scores.append(y_truth)
            predictions.append(prediction)

            # save the model predictions and scores to be aggregated later
            np.save(os.path.join(logging_dir, "{}_solutions.npy".format(run_id)), torch.stack(solutions, dim=1).detach().cpu().numpy())
            np.save(os.path.join(logging_dir, "{}_scores.npy".format(run_id)), torch.stack(scores, dim=1).detach().cpu().numpy())
            np.save(os.path.join(logging_dir, "{}_predictions.npy".format(run_id)), torch.stack(predictions, dim=1).detach().cpu().numpy())