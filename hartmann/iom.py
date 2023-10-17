from models.forward_model import build_forward_model, build_oracle, build_discriminator_model, build_rep_model
from models.iom_model import IOMModel
from models.oracle_model import HartmannOracle
import torch
from scipy.stats import spearmanr
import numpy as np
import os
import json
import time
from utils.logger import Logger
from utils.data import HartmannTask, build_pipeline

import copy
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
        evaluation_samples[0] * evaluation_samples[1]
    )
    logging_filename = '%s_info.log' % run_id
    return logging_filename, run_id


def iom(**config):
    logging_dir=config.get('logging_dir')
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
    forward_model_train_size=config.get('forward_model_train_size')
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
    oracle_val_size=config.get('oracle_val_size')
    oracle_epochs=config.get('oracle_epochs')
    oracle_load=config.get('oracle_load')
    evaluation_samples=config.get('evaluation_samples')
    fast=config.get('fast')

    latent_space_size=config.get('latent_space_size')
    rep_model_activations=config.get('rep_model_activations')
    rep_model_lr=config.get('rep_model_lr')
    rep_model_hidden_size=config.get('rep_model_hidden_size')
    noise_input=config.get('noise_input')
    mmd_param=config.get('mmd_param')
    seed=config.get('seed', None)

    # create the logger and export the experiment parameters
    logging_filename, run_id = _get_log_name(**config)
    logger = Logger(logging_dir, __name__, logging_filename)
    with open(os.path.join(logging_dir, "%s_params.json" % run_id), "w") as f:
        json.dump(config, f, indent=4)

    # initiate the task
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
    logger.logger.info(f"========== Foward Model ==========")
    # representation network
    rep_model = build_rep_model(
          input_shape,
          output_shape=latent_space_size,
          activations=rep_model_activations,
          hidden_size=rep_model_hidden_size
    ).to(device)
    
    forward_model = build_forward_model(
        input_shape=latent_space_size,
        output_shape=output_shape,
        activations=forward_model_activations,
        hidden=forward_model_hidden_size,
        dropout=forward_model_dropout_p,
        final_tanh=forward_model_final_tanh
    ).to(device)

    discriminator_model = build_discriminator_model(
        input_shape=latent_space_size
    ).to(device)
    logger.logger.info("Networks created at device: {}".format(device))

    particle_lr = particle_lr * np.sqrt(np.prod(input_shape))

    # indices = torch.topk(-y[:, 0], k=evaluation_samples)[1]
    indices = torch.topk(-y[:, 0], k=forward_model_batch_size)[1]
    initial_x = torch.index_select(x, index=indices, dim=0)
    initial_y = torch.index_select(y, index=indices, dim=0)
    xt = initial_x

    # create a trainer for IOM
    trainer = IOMModel(
        g=initial_x,
        discriminator_model=discriminator_model,
        mmd_param=mmd_param,
        rep_model=rep_model,
        rep_model_lr=rep_model_lr,
        forward_model=forward_model,
        forward_model_opt=torch.optim.Adam,
        forward_model_lr=forward_model_lr,
        alpha=forward_model_alpha,
        alpha_opt=torch.optim.Adam,
        alpha_lr=forward_model_alpha_lr,
        overestimation_limit=forward_model_overestimation_limit,
        opt_limit=opt_limit,
        particle_lr=particle_lr,
        noise_std=forward_model_noise_std,
        particle_gradient_steps=particle_train_gradient_steps,
        entropy_coefficient=particle_entropy_coefficient,
        oracle=HartmannOracle,
        task=task,
        model_dir=model_dir,
        model_load=forward_model_load
    )
    logger.logger.info("Trainer created at device: {}".format(device))

    train_data, validate_data = build_pipeline(
        x=x, y=y,
        batch_size=forward_model_batch_size,
        val_size=forward_model_val_size,
        train_size=forward_model_train_size
    )

    if seed is not None:
        np.random.seed(seed)

    logger.logger.info("Start forward model training ...")

    # launch the training
    trainer.launch(train_data, validate_data, logger, forward_model_epochs)

    # ------------------------------------------------------------
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


    logger.logger.info('Step [{}/{}]'.format(0, 1 + particle_evaluate_gradient_steps))
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
    
    for step in range(1, 1 + particle_evaluate_gradient_steps):
        logger.logger.info('Step [{}/{}]'.format(step, particle_evaluate_gradient_steps))
        
        # update the set of solution particles
        xt = trainer.optimize(xt, 1)
        final_xt = trainer.optimize(xt, particle_evaluate_gradient_steps)

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
            with torch.no_grad():
                y_truth = HartmannOracle(solution)
            xt_rep = rep_model(xt)
            xt_rep = xt_rep/(torch.sqrt(torch.sum(xt_rep**2, dim=-1, keepdim=True) + 1e-6) + 1e-6)
            prediction = forward_model(xt_rep)
            final_xt_rep = rep_model(final_xt)
            final_xt_rep = final_xt_rep/(torch.sqrt(torch.sum(final_xt_rep**2, dim=-1, keepdim=True) + 1e-6) + 1e-6)
            final_prediction = forward_model(final_xt_rep)

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
            rep = rep_model(x)
            rep = rep/(torch.sqrt(torch.sum(rep**2, dim=-1, keepdim=True) + 1e-6) + 1e-6)
            pred = forward_model(rep)
            if normalize_xs: x = task.denormalize_x(x)
            if normalize_ys: pred = task.denormalize_y(pred)
            pred_x = torch.concat([pred_x, x], dim=0)
            pred_y = torch.concat([pred_y, pred], dim=0)
    np.save(os.path.join(logging_dir, "{}_pred_x.npy".format(run_id)), pred_x.detach().cpu().numpy())
    np.save(os.path.join(logging_dir, "{}_pred_y.npy".format(run_id)), pred_y.detach().cpu().numpy())


