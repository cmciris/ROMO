from models.forward_model import build_forward_model, build_oracle, ROMO, DimensionROMO
from models.gradient_ascent import GradientAscent
from models.romo_model import RetrievalEnhancedMBO
from models.oracle_model import MaximumLikelihoodModel, OracleModel
import torch
from scipy.stats import spearmanr
import numpy as np
import os
import json
import time
from utils.logger import Logger
from utils.data import StaticGraphTask, build_pipeline, build_rim_pipeline

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
    particle_steps = kwargs.get('particle_gradient_steps')
    particle_ent_coefficient = kwargs.get('particle_entropy_coefficient')
    forward_hidden = kwargs.get('forward_model_hidden_size')
    forward_lr = kwargs.get('forward_model_lr')
    forward_noise_std = kwargs.get('forward_model_noise_std')
    forward_batch = kwargs.get('forward_model_batch_size')
    forward_epochs = kwargs.get('forward_model_epochs')
    evaluation_samples = kwargs.get('evaluation_samples')
    run_id = '%s' % (time.strftime('%m%d%H%M%S'))
    run_prefix = '%s_particle_%g_%d_%g_forward_%d_%g_%g_%d_%d_samples_%d' % (
        exp,
        particle_lr, particle_steps, particle_ent_coefficient,
        forward_hidden, forward_lr, forward_noise_std, forward_batch, forward_epochs,
        evaluation_samples
    )
    logging_filename = '%s_info.log' % run_id
    return logging_filename, run_id


def romo(**config):
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
    dimension_oriented=config.get('dimension_oriented')
    size_retrieval_set=config.get('size_retrieval_set')
    retrieval_method=config.get('retrieval_method')
    distil_grad=config.get('distil_grad')
    aggr_grad=config.get('aggr_grad')
    mse_loss_weight=config.get('mse_loss_weight')
    quantile_loss_weight=config.get('quantile_loss_weight')
    quantile_loss_param=config.get('quantile_loss_param')
    particle_lr=config.get('particle_lr')
    particle_gradient_steps=config.get('particle_gradient_steps')
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
    forward_model_train_size=config.get('forward_model_train_size')
    forward_model_pool_size=config.get('forward_model_pool_size')
    forward_model_epochs=config.get('forward_model_epochs')
    forward_model_load=config.get('forward_model_load')
    forward_model_aggregation_method=config.get('forward_model_aggregation_method')
    forward_model_weights=config.get('forward_model_weights')
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

    # -----------------------------------------------------------
    logger.logger.info(f"========== Forward Model ==========")
    # compute the normalized learning rate of the model
    particle_lr = particle_lr * np.sqrt(np.prod(input_shape))
    
    # make a neural network to predict scores
    if dimension_oriented:
        forward_model = DimensionROMO(
            input_shape,
            output_shape=output_shape,
            opt_channel=task_opt_channel,
            activations=forward_model_activations,
            hidden=forward_model_hidden_size,
            dropout=forward_model_dropout_p,
            final_tanh=forward_model_final_tanh,
            aggregation_method=forward_model_aggregation_method,
            weights=forward_model_weights
        ).to(device)
    else:
        forward_model = ROMO(
            input_shape,
            output_shape=output_shape,
            activations=forward_model_activations,
            hidden=forward_model_hidden_size,
            dropout=forward_model_dropout_p,
            final_tanh=forward_model_final_tanh,
            aggregation_method=forward_model_aggregation_method,
            weights=forward_model_weights
        ).to(device)

    # make a trainer for the forward model
    trainer = RetrievalEnhancedMBO(
        forward_model=forward_model,
        forward_model_optim=torch.optim.Adam,
        forward_model_lr=forward_model_lr,
        alpha=forward_model_alpha,
        alpha_optim=torch.optim.Adam,
        alpha_lr=forward_model_alpha_lr,
        overestimation_limit=forward_model_overestimation_limit,
        size_retrieval_set=size_retrieval_set,
        distil_grad=distil_grad,
        aggr_grad=aggr_grad,
        opt_limit=opt_limit,
        particle_lr=particle_lr,
        noise_std=forward_model_noise_std,
        entropy_coefficient=particle_entropy_coefficient,
        mse_loss_weight=mse_loss_weight,
        quantile_loss_weight=quantile_loss_weight,
        quantile_loss_param=quantile_loss_param,
        retrieval_method=retrieval_method,
        model_dir=model_dir,
        model_load=forward_model_load,
    ).to(device)
    logger.logger.info("Model created at device: {}".format(device))

    # create a data set
    train_data, validate_data, pool_data = build_rim_pipeline(
        x=x, aux=aux, y=y,
        batch_size=forward_model_batch_size,
        val_size=forward_model_val_size,
        train_size=forward_model_train_size,
        pool_size=forward_model_pool_size
    )
    # pdb.set_trace()
    logger.logger.info("Start forward model training ...")
    # train the forward model
    trainer.launch(train_data, validate_data, pool_data, logger, forward_model_epochs)
    
    # -----------------------------------------------------------
    logger.logger.info(f"========== Offline Evaluation ==========")
    # pdb.set_trace()
    logger.logger.info("Evaluating {}".format('quickly and only log once!' if fast else 'now!'))
    # select the worst k initial designs from the dataset
    indices = torch.topk(-y[:, 0], k=evaluation_samples)[1]
    initial_x = torch.index_select(x, index=indices, dim=0)
    initial_y = torch.index_select(y, index=indices, dim=0)
    xt = initial_x

    logger.logger.info('Step [{}/{}]'.format(0, 1 + particle_gradient_steps))
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
    
    for step in range(1, 1 + particle_gradient_steps):
        logger.logger.info('Step [{}/{}]'.format(step, particle_gradient_steps))
        
        # update the set of solution particles
        xt = trainer.optimize(xt, 1, pool_data)
        final_xt = trainer.optimize(xt, particle_gradient_steps, pool_data)

        if not fast or step == particle_gradient_steps:

            solution = xt
            if task.is_discrete and in_latent_space:
                pass
                
            if task.is_normalize_x:
                solution_to_save = task.denormalize_x(solution)
            else:
                solution_to_save = solution

            np.save(os.path.join(logging_dir, "{}_solution.npy".format(run_id)), solution_to_save.detach().cpu().numpy())
                
            # evaluate the solutions found by the model
            with torch.no_grad():
                score, _ = oracle(solution)
            retrieval_set = trainer.search_engine(xt, pool_data)
            prediction, _ = forward_model(xt, retrieval_set)
            final_retrieval_set = trainer.search_engine(final_xt, pool_data)
            final_prediction, _ = forward_model(final_xt, final_retrieval_set)

            if normalize_ys:
                score = task.denormalize_y(score)
                prediction = task.denormalize_y(prediction)
                final_prediction = task.denormalize_y(final_prediction)

            # record the prediction and score to the logger
            logger.record(f"score", score, step, percentile=True)
            logger.record(f"score", score, step)
            logger.record(f"solver/model_to_real", torch.tensor(spearmanr(prediction.detach().cpu().numpy()[:, 0], score.detach().cpu().numpy()[:, 0]).correlation), step)
            logger.record(f"solver/distance", torch.linalg.norm(xt - initial_x), step)
            logger.record(f"solver/prediction", prediction, step, percentile=True)
            logger.record(f"solver/prediction", prediction, step)
            logger.record(f"solver/model_overestimation", final_prediction - prediction, step)
            logger.record(f"solver/overestimation", prediction - score, step)
        
        if not fast:
            
            solutions.append(solution_to_save)
            scores.append(score)
            predictions.append(prediction)

            # save the model predictions and scores to be aggregated later
            np.save(os.path.join(logging_dir, "{}_solutions.npy".format(run_id)), torch.stack(solutions, dim=1).detach().cpu().numpy())
            np.save(os.path.join(logging_dir, "{}_scores.npy".format(run_id)), torch.stack(scores, dim=1).detach().cpu().numpy())
            np.save(os.path.join(logging_dir, "{}_predictions.npy".format(run_id)), torch.stack(predictions, dim=1).detach().cpu().numpy())
