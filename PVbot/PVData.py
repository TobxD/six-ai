import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule

from omegaconf.dictconfig import DictConfig
import random

import logging

logger = logging.getLogger(__file__)

def prepareInput(board, toMove):
    ownBoard = [[float(x==toMove) for x in line] for line in board]
    otherBoard = [[float(x==3-toMove) for x in line] for line in board]
    return [torch.FloatTensor(ownBoard), torch.FloatTensor(otherBoard)]

def prepareOutput (y_policy, result, size=10):
    policy = torch.zeros(size*size)
    for ((y,x), prob) in y_policy:
        policy[y*size+x] = prob
    return (policy, result)

class PVData(LightningDataModule):
    def __init__(self, games_list):
        super().__init__()
        self.games_list = games_list

    def prepare_data(self, train_val_split):
        tensor_data = [[(torch.stack(x, dim=0), torch.FloatTensor(y[0]), torch.FloatTensor([y[1]])) for (x,y) in games] for games in self.games_list]
        train_data, val_data = [], []
        for games in tensor_data:
            train_cnt = min(int(train_val_split*len(games))+1, len(games)-1)
            train_data.append(games[:train_cnt])
            val_data.append(games[train_cnt:])
        self.train_data = [item for sublist in train_data for item in sublist]
        self.val_data = [item for sublist in val_data for item in sublist]
        logger.info(f"train/val data samples: {len(self.train_data)}/{len(self.val_data)}")
        random.shuffle(self.train_data)
        random.shuffle(self.val_data)

    def train_dataloader(self, dataloader_conf: DictConfig):
        return DataLoader(self.train_data, shuffle=True, **dataloader_conf)

    def val_dataloader(self, dataloader_conf: DictConfig):
        return DataLoader(self.val_data, shuffle=False, **dataloader_conf)