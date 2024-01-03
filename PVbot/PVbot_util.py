import json
import logging
from pprint import pprint
import util
#sys.path.append('.')
from datetime import datetime
from pathlib import Path

import torch

from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from PVbot import PVData, PVnet

logger = logging.getLogger(__file__)

def trainModel(model, trainer, dataloader, dataloader_conf: DictConfig, save_path):
    trainer.fit(model, train_dataloaders=dataloader.train_dataloader(dataloader_conf), val_dataloaders=dataloader.val_dataloader(dataloader_conf))
    trainer.save_checkpoint(util.toPath("/models/net_{date}.ckpt".format(date=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
    trainer.save_checkpoint(util.toPath("/models/latest.ckpt"))
    trainer.save_checkpoint(util.toPath("models/model.ckpt"))
    trainer.save_checkpoint(util.toPath(save_path))

def readDataWithPolicy(filename):
    data = []
    with open(util.toPath(filename), "r") as f:
        lines = f.readlines()
        logger.info(f"Lines in data file: {len(lines)}")
        lines = [line[:-1] for line in lines]
        for i in range(0, len(lines), 2):
            board, toMove, y_policy = json.loads(lines[i])
            result = int(lines[i+1])
            data.append((PVData.prepareInput(board, toMove), PVData.prepareOutput(y_policy, float(result) if toMove == 2 else float(-result))))
    return data

# augment to include 4 symmetries
def augmentData(positions):
    def mirror1(array2d):
        return torch.flip(array2d, dims=(0,1))
    def mirror2(array2d):
        return torch.transpose(array2d, 0, 1)

    newData = []
    def addPosition(pos, transforms):
        ((own_board, other_board), (policy, res)) = pos
        policy = policy.view((10, 10))
        for transform in transforms:
            own_board = transform(own_board)
            other_board = transform(other_board)
            policy = transform(policy)
        policy = policy.reshape(-1)
        newData.append(((own_board, other_board), (policy, res)))

    for pos in positions:
        addPosition(pos, [])
        addPosition(pos, [mirror1])
        addPosition(pos, [mirror2])
        addPosition(pos, [mirror1, mirror2])
    return newData

def getDataloader(data_cfg):
    if type(data_cfg.train_data_path) == str:
        data_cfg.train_data_path = [data_cfg.train_data_path]
    data = [position for path in data_cfg.train_data_path for position in readDataWithPolicy(path)]
    data = augmentData(data)
    dataloader = PVData.PVData(data)
    dataloader.prepare_data(data_cfg.train_val_split)
    return dataloader

def training(cfg: DictConfig):
    #print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    network = PVnet.getModel(cfg, cfg.general_train.input_model_path)
    #print(network)

    #trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer_logger = TensorBoardLogger(util.toPath("/lightning_logs"))
    trainer = Trainer(**cfg.pl_trainer, logger=trainer_logger, log_every_n_steps=5)
    #trainer.fit(network,data)

    #seed_everything(42, workers=True)
    #testNet(network)
    trainModel(network, trainer, getDataloader(cfg.data), cfg.data.train_dataloader_conf, cfg.general_train.output_model_path)
    #testNet(network)
