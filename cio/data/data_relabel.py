import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from models.forward_model import build_forward_model, build_oracle
from models.gradient_ascent import GradientAscent
from models.oracle_model import MaximumLikelihoodModel, OracleModel
import torch
from scipy.stats import spearmanr
import numpy as np
import json
import time
from utils.logger import Logger
from utils.data import StaticGraphTask, CIODataSet
from torch.utils.data import DataLoader

import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    normalize_ys = True
    normalize_xs = True
    in_latent_space = False
    oracle_ensembles = 5
    oracle_dir = 'checkpoints/oracle/fc'
    batch_size = 2048
    epochs = 200
    run_id = '%s' % (time.strftime('%m%d%H%M%S'))
    logger = Logger('./relabel', __name__, f'{run_id}_relabel.log')

    task = StaticGraphTask(data_dir='./',
                           score_scale=True,
                           score_range='(0, 100)')

    if normalize_ys:
        task.map_normalize_y()
        task.map_normalize_aux()
    if task.is_discrete and not in_latent_space:
        task.map_to_logits()
    if normalize_xs:
        task.map_normalize_x()
    
    initial_x = torch.tensor(task.x, dtype=torch.float32, device=device)
    initial_aux = torch.tensor(task.aux, dtype=torch.float32, device=device)
    initial_y = torch.tensor(task.y, dtype=torch.float32, device=device)

    if task.is_discrete and in_latent_space:
        pass

    input_shape = initial_x.shape[1:]
    output_shape = initial_y.shape[1:]

    oracle_models = [
        build_oracle(
        input_shape=input_shape,
        output_shape=output_shape,
        activations=['nn.ReLU', 'nn.ReLU'],
        hidden=2048,
        dropout=0.0,
        final_tanh=False
        ).to(device) for _ in range(oracle_ensembles)
    ]

    oracle_trainers = [
        MaximumLikelihoodModel(
        oracle=model,
        oracle_optim=torch.optim.Adam,
        oracle_lr=0.0003,
        logger_prefix=f"oracle{i+1}",
        noise_std=0.0,
        model_dir=os.path.join('../',oracle_dir),
        model_load=True
        ).to(device) for i, model in enumerate(oracle_models)
    ]

    # create DataLoader

    # shuffle
    # indices = np.arange(initial_x.shape[0])
    # np.random.shuffle(indices)
    # initial_x = initial_x[indices]
    # initial_aux = initial_aux[indices]
    # initial_y = initial_y[indices]

    relabel_inputs = [initial_x, initial_aux, initial_y]
    relabel_dataset = CIODataSet(relabel_inputs)
    relabel_dataloader = DataLoader(relabel_dataset, batch_size=batch_size, shuffle=False)

    for i, t in enumerate(oracle_trainers):
        logger.logger.info(f"---------- Oracle {i+1} ----------")
        t.launch(train_data=None, validate_data=None, logger=logger, epochs=epochs)
    
    oracle = OracleModel(oracle_models)
    aux_relabeled = []
    for infer_step, (x, aux, y) in enumerate(relabel_dataloader):
        logger.logger.info(f"inference step {infer_step}")
        aux_prediction, _ = oracle(x)
        aux_relabeled.append(aux_prediction)
    logger.logger.info("done")
    aux_relabeled = torch.cat(aux_relabeled, dim=0)
    assert aux_relabeled.shape[0] == initial_aux.shape[0]
    size = aux_relabeled.shape[0]
    
    if normalize_ys:
        aux_relabeled = task.denormalize_aux(aux_relabeled).detach().cpu().numpy()
    y_relabeled = - np.sum(np.var(aux_relabeled.reshape(size, -1, 2), axis=-2), axis=-1, keepdims=True)
    x_relabeled = task.denormalize_x(initial_x).detach().cpu().numpy()

    # pdb.set_trace()
    logger.logger.info(f"x: {x_relabeled.shape}, aux: {aux_relabeled.shape}, y: {y_relabeled.shape}")
    np.save('./relabel/x.npy', x_relabeled)
    np.save('./relabel/aux.npy', aux_relabeled)
    np.save('./relabel/y.npy', y_relabeled)


if __name__ == "__main__":
    main()