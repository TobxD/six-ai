import json
import logging
import util
#sys.path.append('.')
from datetime import datetime
from pathlib import Path

from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from PVbot import PVData, PVnet

logger = logging.getLogger(__name__)

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
        print(f"Lines in data file: {len(lines)}")
        lines = [line[:-1] for line in lines]
        for i in range(0, len(lines), 2):
            board, toMove, y_policy = json.loads(lines[i])
            result = int(lines[i+1])
            data.append((PVData.prepareInput(board, toMove), PVData.prepareOutput(y_policy, float(result) if toMove == 2 else float(-result))))
    return data

def getDataloader(data_cfg):
    if type(data_cfg.train_data_path) == str:
        data_cfg.train_data_path = [data_cfg.train_data_path]
    data = [position for path in data_cfg.train_data_path for position in readDataWithPolicy(path)]
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
