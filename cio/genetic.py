from models.forward_model import build_forward_model, build_oracle
from models.coms_model import ConservativeObjectiveModel
from models.oracle_model import MaximumLikelihoodModel, GCNConvMaximumLikelihoodModel, OracleModel
from models.genetic import UniformCrossover, UniformMutate, NormalMutate
import torch
from scipy.stats import spearmanr
import numpy as np
import os
import copy
import json
import time
from utils.logger import Logger
from utils.data import StaticGraphTask, build_pipeline

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
    particle_lr = kwargs.get('particle_lr')
    particle_train_steps = kwargs.get('particle_train_gradient_steps')
    particle_ent_coefficient = kwargs.get('particle_entropy_coefficient')
    forward_hidden = kwargs.get('forward_model_hidden_size')
    forward_lr = kwargs.get('forward_model_lr')
    forward_alpha_lr = kwargs.get('forward_model_alpha_lr')
    forward_limit = kwargs.get('forward_model_overestimation_limit')
    forward_noise_std = kwargs.get('forward_model_noise_std')
    forward_batch = kwargs.get('forward_model_batch_size')
    forward_epochs = kwargs.get('forward_model_epochs')
    evaluation_samples = kwargs.get('evaluation_samples')
    run_id = '%s' % (time.strftime('%m%d%H%M%S'))
    run_prefix = '%s_particle_%g_%d_%g_forward_%d_%g_%g_%g_%g_%d_%d_samples_%d' % (
        exp,
        particle_lr, particle_train_steps, particle_ent_coefficient,
        forward_hidden, forward_lr, forward_alpha_lr, forward_limit, forward_noise_std, forward_batch, forward_epochs,
        evaluation_samples
    )
    logging_filename = '%s_info.log' % run_id
    return logging_filename, run_id


def genetic(**config):
    logging_dir=os.path.join(config.get('logging_dir'), config.get('exp'))
    data_dir=config.get('data_dir')
    model_dir=config.get('model_dir')
    oracle_dir=config.get('oracle_dir')
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
    vae_hidden_size=config.get('vae_hidden_size')
    vae_latent_size=config.get('vae_latent_size')
    vae_activation=config.get('vae_activation')
    vae_kernel_size=config.get('vae_kernel_size')
    vae_num_blocks=config.get('vae_num_blocks')
    vae_lr=config.get('vae_lr')
    vae_beta=config.get('vae_beta')
    vae_batch_size=config.get('vae_batch_size')
    vae_val_size=config.get('vae_val_size')
    vae_epochs=config.get('vae_epochs')
    particle_lr=config.get('particle_lr')
    particle_train_gradient_steps=config.get('particle_train_gradient_steps')
    particle_evaluate_gradient_steps=config.get('particle_evaluate_gradient_steps')
    particle_entropy_coefficient=config.get('particle_entropy_coefficient')
    forward_model_activations=config.get('forward_model_activations')
    forward_model_hidden_size=config.get('forward_model_hidden_size')
    forward_model_dropout_p=config.get('forward_model_dropout_p')
    forward_model_final_tanh=config.get('forward_model_final_tanh')
    forward_model_lr=config.get('forward_model_lr')
    forward_model_alpha=config.get('forward_model_alpha')
    forward_model_alpha_lr=config.get('forward_model_alpha_lr')
    forward_model_overestimation_limit=config.get('forward_model_overestimation_limit')
    forward_model_noise_std=config.get('forward_model_noise_std')
    forward_model_batch_size=config.get('forward_model_batch_size')
    forward_model_val_size=config.get('forward_model_val_size')
    forward_model_epochs=config.get('forward_model_epochs')
    forward_model_load=config.get('forward_model_load')
    oracle_ensembles=config.get('oracle_ensembles')
    oracle_activations=config.get('oracle_activations')
    oracle_hidden_size=config.get('oracle_hidden_size')
    oracle_dropout_p=config.get('oracle_dropout_p')
    oracle_final_tanh=config.get('oracle_final_tanh')
    oracle_lr=config.get('oracle_lr')
    oracle_noise_std=config.get('oracle_noise_std')
    oracle_batch_size=config.get('oracle_batch_size')
    oracle_train_size=config.get('oracle_train_size')
    oracle_val_size=config.get('oracle_val_size')
    oracle_epochs=config.get('oracle_epochs')
    oracle_load=config.get('oracle_load')
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
    task = StaticGraphTask(data_dir,
                           score_scale=task_score_scale,
                           score_range=task_score_range,relabel=task_relabel,
                           dataset_kwargs=dict(max_samples=task_max_samples, distribution=task_distribution,
                           opt_channel=task_opt_channel,
                           opt_ub=task_opt_ub,
                           opt_lb=task_opt_lb))

    if normalize_ys:
        task.map_normalize_y()
        task.map_normalize_aux()
    if task.is_discrete and not in_latent_space:
        task.map_to_logits()
    if normalize_xs:
        task.map_normalize_x()

    # save the initial dataset statistics for safe keeping
    x = torch.tensor(task.x, dtype=torch.float32, device=device)
    aux = torch.tensor(task.aux, dtype=torch.float32, device=device)
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

    # -----------------------------------------------------------
    logger.logger.info(f"========== Oracle ==========")
    # make an ensemble neural networks to predict scores as the oracle
    oracle_models = [
        build_oracle(
        input_shape=input_shape,
        output_shape=output_shape,
        activations=oracle_activations,
        hidden=oracle_hidden_size,
        dropout=oracle_dropout_p,
        final_tanh=oracle_final_tanh
        ).to(device) for _ in range(oracle_ensembles)
    ]

    # create a ensemble trainers for the oracle models
    oracle_trainers = [
        MaximumLikelihoodModel(
        oracle=model,
        oracle_optim=torch.optim.Adam,
        oracle_lr=oracle_lr,
        logger_prefix=f"oracle{i+1}",
        noise_std=oracle_noise_std,
        model_dir=oracle_dir,
        model_load=oracle_load
        ).to(device) for i, model in enumerate(oracle_models)
    ]
    logger.logger.info("Oracle ensembles {} created at device: {}".format(oracle_ensembles, device))
    
    # create bootstraps of a dataset
    oracle_train_datas, validate_data = build_pipeline(
        x=x, aux=aux, y=y,
        batch_size=oracle_batch_size,
        train_size=oracle_train_size,
        val_size=oracle_val_size,
        bootstraps=oracle_ensembles
    )
    # pdb.set_trace()
    logger.logger.info("Start oracle training ...")
    # train the oracle ensembles
    for i, t in enumerate(oracle_trainers):
        logger.logger.info(f"---------- Oracle {i+1} ----------")
        t.launch(oracle_train_datas[i], validate_data, logger, oracle_epochs)
    
    oracle = OracleModel(oracle_models)
    
    # pdb.set_trace()
    logger.logger.info("Evaluating {}".format('quickly and only log once!' if fast else 'now!'))
    # select the worst k initial designs from the dataset
    indices = torch.topk(-y[:, 0], k=evaluation_samples)[1]
    initial_x = torch.index_select(x, index=indices, dim=0)
    initial_y = torch.index_select(y, index=indices, dim=0)
    xt = copy.deepcopy(initial_x)

    logger.logger.info(f"========== Offline Evaluation ==========")
    logger.logger.info('Step [{}/{}]'.format(0, 1 + particle_evaluate_gradient_steps))
    if not fast:

        scores = []
        predictions = []
        solutions = []

        solution = xt
        if task.is_discrete and in_latent_space:
            pass
        
        with torch.no_grad():
            score, _ = oracle(solution)
        
        if normalize_ys:
            initial_y = task.denormalize_y(initial_y)
            score = task.denormalize_y(score)

        if task.is_normalize_x:
            solution_to_save = task.denormalize_x(solution)
        else:
            solution_to_save = solution
        solutions.append(solution_to_save)  
        
        logger.record(f"dataset_score", initial_y, 0, percentile=True)
        logger.record(f"dataset_score", initial_y, 0)
        logger.record(f"score", score, 0, percentile=True)
        logger.record(f"score", score, 0)
    
    for step in range(1, 1 + particle_evaluate_gradient_steps):
        logger.logger.info('Step [{}/{}]'.format(step, particle_evaluate_gradient_steps))
        
        # update the set of solution particles
        with torch.no_grad():
            score, _ = oracle(xt)
        if normalize_ys:
            score = task.denormalize_y(score)

        _, index = torch.sort(score, dim=0, descending=True)
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
        
        if not fast or step == particle_evaluate_gradient_steps:

            solution = xt
            if task.is_discrete and in_latent_space:
                pass
                
            if task.is_normalize_x:
                solution_to_save = task.denormalize_x(solution)
            else:
                solution_to_save = solution

            np.save(os.path.join(logging_dir, "{}_solution.npy".format(run_id)), solution_to_save.detach().cpu().numpy())
                
            # evaluate the solutions found by the model
            score, _ = oracle(solution)

            if normalize_ys:
                score = task.denormalize_y(score)

            # record the prediction and score to the logger
            logger.record(f"score", score, step, percentile=True)
            logger.record(f"score", score, step)
            logger.record(f"solver/distance", torch.linalg.norm(xt - initial_x), step)
        
        if not fast:
            
            solutions.append(solution_to_save)
            scores.append(score)

            # save the model predictions and scores to be aggregated later
            np.save(os.path.join(logging_dir, "{}_solutions.npy".format(run_id)), torch.stack(solutions, dim=1).detach().cpu().numpy())
            np.save(os.path.join(logging_dir, "{}_scores.npy".format(run_id)), torch.stack(scores, dim=1).detach().cpu().numpy())