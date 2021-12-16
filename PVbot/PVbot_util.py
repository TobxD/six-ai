import json
import logging
#sys.path.append('.')
from datetime import datetime
from pathlib import Path

from hydra.utils import instantiate, to_absolute_path
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

from PVbot import PVData, PVnet

logger = logging.getLogger(__name__)

def trainModel(model, trainer, dataloader, dataloader_conf: DictConfig):
    trainer.fit(model, train_dataloaders=dataloader.train_dataloader(dataloader_conf), val_dataloaders=dataloader.val_dataloader(dataloader_conf))
    trainer.save_checkpoint(to_absolute_path("models/net_{date}.ckpt".format(date=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
    trainer.save_checkpoint(to_absolute_path("models/latest.ckpt"))
    trainer.save_checkpoint(Path("models/model.ckpt"))


def readDataWithPolicy(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        print(f"Lines in data file: {len(lines)}")
        lines = [line[:-1] for line in lines]
        for i in range(0, len(lines), 2):
            #print(i)
            #print(lines[i])
            board, toMove, y_policy = json.loads(lines[i])
            result = int(lines[i+1])
            data.append((PVData.prepareInput(board, toMove), PVData.prepareOutput(y_policy, float(result) if toMove == 2 else float(-result))))
    return data

def getDataloader(datapath):
    data = readDataWithPolicy(to_absolute_path(datapath))
    #pprint(data[0])
    dataloader = PVData.PVData(data, batch_size=1024)
    dataloader.prepare_data()
    return dataloader

def training(cfg: DictConfig):
    print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    #network = getModel(cfg)
    network = PVnet.getModel(new=False, path=to_absolute_path("models/latest.ckpt"), cfg=cfg)
    print(network)

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = Trainer(**cfg.pl_trainer, logger=trainer_logger)
    #trainer.fit(network,data)

    seed_everything(42, workers=True)
    #testNet(network)
    trainModel(network, trainer, getDataloader(cfg.data.train_data_path), cfg.data.train_dataloader_conf)
    #testNet(network)
