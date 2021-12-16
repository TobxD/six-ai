import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule

from omegaconf.dictconfig import DictConfig
import random

def prepareInput(board, toMove):
    ownBoard = [[float(x==toMove) for x in line] for line in board]
    otherBoard = [[float(x==3-toMove) for x in line] for line in board]
    return [ownBoard, otherBoard]

def prepareOutput (y_policy, result, size=10):
    policy = torch.zeros(size*size)
    for ((y,x), prob) in y_policy:
        policy[y*size+x] = prob
    return (policy, result)

class PVData(LightningDataModule):
    def __init__(self, games, batch_size=4):
        super().__init__()
        self.games = games
        self.batch_size = batch_size

    def prepare_data(self):
        tensor_data = [(torch.FloatTensor(x), torch.FloatTensor(y[0]), torch.FloatTensor([y[1]])) for (x,y) in self.games]
        train_cnt = min(int(0.8*len(tensor_data))+1, len(tensor_data)-1)
        # don't shuffle, otherwise validation data familiar to net; here done for illustration purposes
        #random.shuffle(tensor_data)
        self.train_data = tensor_data[:train_cnt]
        self.val_data = tensor_data[train_cnt:]
        print(len(self.train_data), len(self.val_data))
        random.shuffle(self.train_data)
        random.shuffle(self.val_data)
        #self.train_data = self.train_data[:256]
        #self.val_data = self.val_data[:256]

    def train_dataloader(self, dataloader_conf: DictConfig):
        return DataLoader(self.train_data, **dataloader_conf)

    def val_dataloader(self, dataloader_conf: DictConfig):
        return DataLoader(self.val_data, **dataloader_conf)