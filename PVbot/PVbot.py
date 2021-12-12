import json, os, sys, logging, random

sys.path.append('../six-ai')
from datetime import datetime
from typing import Dict

from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
import torch
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from main import simulate

from randomBot import RandomBot

from nnBot import prepareInput
from board import Board
from PVnet import PVnet
from pathlib import Path
import hydra
from hydra.utils import instantiate

from pytorch_lightning import Trainer, LightningDataModule, seed_everything

from util import *

logger = logging.getLogger(__name__)

class PolicyData(LightningDataModule):
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

def getModel(cfg = None, new = True, path = None):
    hparams = {
        #also good: 5e-4
        'lr': 1e-3,
        'reg': 0,
        'channels': 20
    }
    if new:
        return PVnet(train_conf=cfg.train, network_conf=cfg.network_conf)
    else:
        if path == None:
            path = Path("models/latest.ckpt")
        return PVnet.load_from_checkpoint(path, s=SIZE, hparams=hparams)

def trainModel(model, trainer, dataloader, dataloader_conf: DictConfig):
    trainer.fit(model, train_dataloaders=dataloader.train_dataloader(dataloader_conf), val_dataloaders=dataloader.val_dataloader(dataloader_conf))
    trainer.save_checkpoint(Path("models/net_{date}.ckpt".format(date=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
    trainer.save_checkpoint(Path("models/latest.ckpt"))

def prepareOutput (move, result, size=10):
    y_move, x_move = move

    policy = [0.0 for _ in range(size*size)]
    policy[y_move*size+x_move] = 1.0
    
    return (policy, result)

def readDataWithPolicy(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        print(f"Lines in data file: {len(lines)}")
        lines = [line[:-1] for line in lines]
        for i in range(0, len(lines), 2):
            board, move, toMove = json.loads(lines[i])
            result = int(lines[i+1])
            data.append((prepareInput(board, toMove), prepareOutput(move, float(result) if toMove == 1 else float(-result))))
            #print(i)
    return data

def getDataloader(datapath):
    data = readDataWithPolicy(Path(os.path.dirname(__file__)+"/../"+datapath))
    #pprint(data[0])
    dataloader = PolicyData(data, batch_size=1024)
    dataloader.prepare_data()
    return dataloader

def trainModelFromData(model, trainer, datapath, dataloader_conf):
    trainModel(model, trainer, getDataloader(datapath), dataloader_conf)
    #print(getError(model, dataloader.val_dataloader()))

def testNet (net):
    board = Board(10, True)
    posMoves = board.movesAvailable()
    positions = []
    for (y, x) in posMoves:
        if board.wouldWin(1, y, x):
            return (y, x)
        board.move(1, y, x)
        positions.append((prepareInput(board.board, 2), (y,x)))
        board.move(0, y, x)
    tensor_data = [(torch.FloatTensor(x).float(), y) for (x,y) in positions]
    dataloader = DataLoader(tensor_data, batch_size=256)
    print(posMoves)

    for (idx, batch) in enumerate(dataloader):
        print (net(batch[0]))

class PVBot:
    myColor = 1
    otherColor = 2

    def __init__(self, myColor):
        self.myColor = myColor
        self.otherColor = 3-myColor
        self.model = getModel (new=False, path=Path("../../2021-12-12/13-38-35/models/latest.ckpt"))
        self.model.eval()

    def nextMove(self,board:Board):
        print (board, self.myColor)
        X = torch.FloatTensor(prepareInput(board.board,self.myColor)) 
        policy, value = self.model(X.unsqueeze(0))  
        #validMoves = board.movesAvailableAsTensor()
        #validPolicy = policy * torch.flatten(torch.FloatTensor(validMoves))

        movesAvailable = board.movesAvailable()
        policyForSoftmax = []
        policy = policy.tolist()[0]
        #print(policy[0])
        for move in movesAvailable:
            policyForSoftmax.append(policy[board.convert2Dto1Dindex(move)])
        
        policyForSoftmax = torch.softmax(torch.FloatTensor(policyForSoftmax),0)
        #print(policyForSoftmax)

        validPolicy = torch.zeros(board.size*board.size)
        for i, move in enumerate(movesAvailable):
            validPolicy[board.convert2Dto1Dindex(move)] = policyForSoftmax[i]

        #validPolicy = torch.softmax(validPolicy, dim=1)
        #print (validPolicy)
        move1D = torch.argmax(validPolicy).item()
        move2D = board.convert1Dto2Dindex(move1D)
        print (move2D)
        print (torch.amax(validPolicy).item())
        
        return move2D

@hydra.main(config_path="conf", config_name="PVconfig")
def training(cfg: DictConfig):
    print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    network = getModel(cfg)
    print(network)

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = Trainer(**cfg.pl_trainer, logger=trainer_logger)
    #trainer.fit(network,data)

    seed_everything(42, workers=True)
    #testNet(network)
    trainModelFromData(network, trainer, cfg.data.train_data_path, cfg.data.train_dataloader_conf)
    testNet(network)

@hydra.main(config_path="conf", config_name="PVconfig")
def testNet(cfg: DictConfig):
    board = Board(SIZE, startPieces=True)
    player1 = PVBot(1)
    player2 = RandomBot(2, search_winning=True, search_losing=True)
    game = simulate(board, player1, player2)
    print(game[1])

if __name__ == "__main__":
    #data = readDataWithPolicy(Path("data/data.json"))
    #training()
    testNet()