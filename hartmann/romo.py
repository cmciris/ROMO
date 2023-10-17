from models.forward_model import build_forward_model, build_oracle, ROMO, DimensionROMO
from models.gradient_ascent import GradientAscent
from models.romo_model import RetrievalEnhancedMBO
from models.oracle_model import HartmannOracle
import torch
from scipy.stats import spearmanr
import numpy as np
import os
import json
import time
import copy
from utils.logger import Logger
from utils.data import HartmannTask, build_pipeline, build_rim_pipeline

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
        evaluation_samples[0] * evaluation_samples[1]
    )
    logging_filename = '%s_info.log' % run_id
    return logging_filename, run_id


def romo(**config):
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
    evaluation_samples=config.get('evaluation_samples')
    fast=config.get('fast')
    
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
        x=x, y=y,
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

    logger.logger.info('Step [{}/{}]'.format(0, 1 + particle_gradient_steps))
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
                y_truth = HartmannOracle(solution)
            retrieval_set = trainer.search_engine(xt, pool_data)
            prediction, _ = forward_model(xt, retrieval_set)
            final_retrieval_set = trainer.search_engine(final_xt, pool_data)
            final_prediction, _ = forward_model(final_xt, final_retrieval_set)

            if normalize_ys:
                prediction = task.denormalize_y(prediction)
                final_prediction = task.denormalize_y(final_prediction)

            # record the prediction and score to the logger
            logger.record(f"score", y_truth, step, percentile=True)
            logger.record(f"score", y_truth, step)
            logger.record(f"solver/model_to_real", torch.tensor(spearmanr(prediction.detach().cpu().numpy()[:, 0], y_truth.detach().cpu().numpy()[:, 0]).correlation), step)
            logger.record(f"solver/distance", torch.linalg.norm(xt - initial_x), step)
            logger.record(f"solver/prediction", prediction, step, percentile=True)
            logger.record(f"solver/prediction", prediction, step)
            logger.record(f"solver/model_overestimation", final_prediction - prediction, step)
            logger.record(f"solver/overestimation", prediction - y_truth, step)
        
        if not fast:
            
            solutions.append(solution_to_save)
            scores.append(y_truth)
            predictions.append(prediction)

            # save the model predictions and scores to be aggregated later
            np.save(os.path.join(logging_dir, "{}_solutions.npy".format(run_id)), torch.stack(solutions, dim=1).detach().cpu().numpy())
            np.save(os.path.join(logging_dir, "{}_scores.npy".format(run_id)), torch.stack(scores, dim=1).detach().cpu().numpy())
            np.save(os.path.join(logging_dir, "{}_predictions.npy".format(run_id)), torch.stack(predictions, dim=1).detach().cpu().numpy())

    samples = 15
    sublayer = [4,5]
    x0 = np.linspace(0, 1, samples)
    x1 = np.linspace(0, 1, samples)
    x0, x1 = np.meshgrid(x0, x1)
    np.random.seed(42)
    random = np.random.random(samples * samples)
    data = np.zeros((0, 3))
    for i in range(sublayer[0]):
        for j in range(sublayer[1]):
            layer = i * sublayer[1] + j
            x2 = (layer / np.prod(sublayer)) + random * (1 / np.prod(sublayer))
            layer_data = np.concatenate([x0.reshape(-1, 1), x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=-1)
            data = np.concatenate([data, layer_data], axis=0)
    loader, _ = build_pipeline(x=data, y=np.zeros([data.shape[0], 1]), val_size=0, train_size=data.shape[0], batch_size=1024)
    pred_x, pred_y = torch.zeros([0, 3], device=device), torch.zeros([0, 1], device=device)
    forward_model.eval()
    with torch.no_grad():
        for (x, _) in loader:
            x = x.to(device).float()
            retrieval_set = trainer.search_engine(x, pool_data)
            pred, _ = forward_model(x, retrieval_set)
            if normalize_xs: x = task.denormalize_x(x)
            if normalize_ys: pred = task.denormalize_y(pred)
            pred_x = torch.concat([pred_x, x], dim=0)
            pred_y = torch.concat([pred_y, pred], dim=0)
    np.save(os.path.join(logging_dir, "{}_pred_x.npy".format(run_id)), pred_x.detach().cpu().numpy())
    np.save(os.path.join(logging_dir, "{}_pred_y.npy".format(run_id)), pred_y.detach().cpu().numpy())