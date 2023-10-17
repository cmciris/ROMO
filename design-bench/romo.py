from models.forward_model import build_forward_model, ROMO, DimensionRIM
from models.romo_model import RetrievalEnhancedMBO
from models.vae_model import SequentialVAE, VAETrainer
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
    logging_dir=config.get('logging_dir')
    model_dir=config.get('model_dir')
    task_name=config.get('task')
    task_relabel=config.get('task_relabel')
    task_max_samples=config.get('task_max_samples')
    task_distribution=config.get('task_distribution')
    task_opt_channel=config.get('task_opt_channel')
    normalize_ys=config.get('normalize_ys')
    normalize_xs=config.get('normalize_xs')
    in_latent_space=config.get('in_latent_space', False)
    vae_hidden_size=config.get('vae_hidden_size', 64)
    vae_latent_size=config.get('vae_latent_size', 256)
    vae_activation=config.get('vae_activation', 'nn.ReLU')
    vae_kernel_size=config.get('vae_kernel_size', 3)
    vae_num_blocks=config.get('vae_num_blocks', 4)
    vae_lr=config.get('vae_lr', 0.0003)
    vae_beta=config.get('vae_beta', 1.0)
    vae_batch_size=config.get('vae_batch_size', 32)
    vae_val_size=config.get('vae_val_size', 200)
    vae_epochs=config.get('vae_epochs', 10)
    dimension_oriented=config.get('dimension_oriented')
    size_retrieval_set=config.get('size_retrieval_set')
    retrieval_method=config.get('retrieval_method')
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
    forward_model_train_ratio=config.get('forward_model_train_ratio')
    forward_model_epochs=config.get('forward_model_epochs')
    forward_model_aggregation_method=config.get('forward_model_aggregation_method')
    forward_model_load=config.get('forward_model_load')
    forward_model_weights=config.get('forward_model_weights')
    mse_loss_weight=config.get('mse_loss_weight')
    evaluation_samples=config.get('evaluation_samples')
    fast=config.get('fast')
    
    # create the logger and export the experiment parameters
    logging_filename, run_id = _get_log_name(**config)
    logger = Logger(logging_dir, __name__, logging_filename)
    with open(os.path.join(logging_dir, "%s_params.json" % run_id), "w") as f:
        json.dump(config, f, indent=4)
    
    # create a model-based optimization task
    task = StaticGraphTask(task_name,
                           relabel=task_relabel,
                           dataset_kwargs=dict(max_samples=task_max_samples, distribution=task_distribution))

    if normalize_ys:
        task.map_normalize_y()
    if task.is_discrete and not in_latent_space:
        task.map_to_logits()
    if normalize_xs:
        task.map_normalize_x()

    # save the initial dataset statistics for safe keeping
    x = torch.tensor(task.x, dtype=torch.float32, device=device)
    if task_relabel and 'Hopper' in task_name:
        y = torch.tensor(task.y_relabel, dtype=torch.float32, device=device)
    else:
        y = torch.tensor(task.y, dtype=torch.float32, device=device)

    if task.is_discrete and in_latent_space:
        # vae
        vae_model = SequentialVAE(
            task, hidden_size=vae_hidden_size,
            latent_size=vae_latent_size, activation=vae_activation,
            kernel_size=vae_kernel_size, num_blocks=vae_num_blocks
        ).to(device)

        vae_trainer = VAETrainer(
            vae_model, optim=torch.optim.Adam,
            lr=vae_lr, beta=vae_beta
        ).to(device)

        # create the training task and logger
        train_data, val_data = build_pipeline(
            x=x, y=y, batch_size=vae_batch_size,
            val_size=vae_val_size
        )

        # estimate the number of training steps per epoch
        vae_trainer.launch(train_data, val_data, logger, vae_epochs)

        # map the x values to latent space
        x = vae_model.encoder_cnn(x)[0]

        mean = torch.mean(x, dim=0, keepdim=True)
        standard_dev = torch.std(x - mean, dim=0, keepdim=True)
        x = (x - mean) / standard_dev

    logger.logger.info("Task created: {}".format(task_name))
    
    input_shape = x.shape[1:]
    if task_opt_channel < 1:
        task_opt_channel = int(task_opt_channel * input_shape[0])
        print('x:', x.shape, 'y:', y.shape, 'task_opt_channel:', task_opt_channel)
    logger.logger.info("x: {}, y: {}, task_opt_channel: {}".format(x.shape, y.shape, task_opt_channel))

    # -----------------------------------------------------------
    logger.logger.info(f"========== Forward Model ==========")
    # compute the normalized learning rate of the model
    particle_lr = particle_lr * np.sqrt(np.prod(input_shape))
    
    # make a neural network to predict scores
    if dimension_oriented:
        forward_model = DimensionRIM(
            input_shape,
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
        opt_limit=task_opt_channel,
        particle_lr=particle_lr,
        noise_std=forward_model_noise_std,
        entropy_coefficient=particle_entropy_coefficient,
        mse_loss_weight=mse_loss_weight,
        retrieval_method=retrieval_method,
        model_dir=model_dir,
        model_load=forward_model_load
    ).to(device)
    logger.logger.info("Model created at device: {}".format(device))

    # create a data set
    train_data, validate_data, pool_data = build_rim_pipeline(
        x=x, y=y,
        batch_size=forward_model_batch_size,
        val_size=forward_model_val_size,
        train_ratio=forward_model_train_ratio
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
            # vae decode
            solution = solution * standard_dev + mean
            logits = vae_model.decoder_cnn(solution)
            solution = torch.argmax(logits, dim=2)
        
        score = task.predict(solution)
        
        if normalize_ys:
            initial_y = task.denormalize_y(initial_y)
            score = task.denormalize_y(score)
        
        
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
                # vae decode
                solution = solution * standard_dev + mean
                logits = vae_model.decoder_cnn(solution)
                solution = torch.argmax(logits, dim=2)
                
            if normalize_xs:
                solution_to_save = task.denormalize_x(solution)

            np.save(os.path.join(logging_dir, "{}_solution.npy".format(run_id)), solution_to_save.detach().cpu().numpy())
                
            # evaluate the solutions found by the model
            score = task.predict(solution)
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
