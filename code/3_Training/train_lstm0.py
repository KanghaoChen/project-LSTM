import pdb
import pandas as pd

from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from fastai.learner import Learner, DataLoaders, DataLoader
from fastai.callback.data import WeightedDL
from fastai.metrics import accuracy, Recall, Precision, BalancedAccuracy, error_rate, F1Score, RocAuc, RocAucBinary
from fastai.callback.tensorboard import TensorBoardCallback

from torch.utils.tensorboard import SummaryWriter

from dataset import FiresDataset
from models import lstm0Classifier


@hydra.main(config_path='configs/', config_name='train_lstm0')
def train(config: DictConfig) -> None:

    # Create datasets and dataloaders
    train_dataset, test_dataset = FiresDataset.create_datasets_from_dir(**config.dataset)
    train_weights = np.ones(len(train_dataset))
    train_weights[train_dataset.labels] = config.fire_weight
    train_loader = WeightedDL(
        train_dataset, bs=config.batch_size, wgts=train_weights, shuffle=True, drop_last=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(test_dataset, bs=config.batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    data = DataLoaders(train_loader, test_loader, device=config.device)
    print(train_dataset.df)

    # Create model
    modellstm0 = instantiate(config.model).to(config.device)

    # Create learner and run
    learner = Learner(data, modellstm0, loss_func=F.cross_entropy, metrics=[accuracy, Recall(), Precision(), BalancedAccuracy(),RocAucBinary()])
    learner.fit(n_epoch=config.n_epoch, lr=config.lr, cbs=[TensorBoardCallback(log_preds=False)])
    
    learner.save('model_output')





if __name__ == "__main__":
    train()

