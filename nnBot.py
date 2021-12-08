import json, datetime
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pprint

from net import ValueNet, GameData
from util import *

def prepareData(board, toMove):
    ownBoard = [[float(x==toMove) for x in line] for line in board]
    otherBoard = [[float(x==3-toMove) for x in line] for line in board]
    return [ownBoard, otherBoard]

def readData(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]
        for i in range(0, len(lines), 2):
            board, toMove = json.loads(lines[i])
            result = int(lines[i+1])
            data.append((prepareData(board, toMove), float(result) if toMove == 1 else float(-result)))
    pprint.pprint(data[0])
    return data

def trainModel(model, dataloader):
    trainer = pl.Trainer(
        weights_summary=None,
        max_epochs=10,
        progress_bar_refresh_rate=25,
        gpus=1
    )
    trainer.fit(model, train_dataloaders=dataloader.train_dataloader(), val_dataloaders=dataloader.val_dataloader())
    trainer.save_checkpoint("models/net_{date:%Y-%m-%d_%H:%M:%S}.ckpt".format(date=datetime.datetime.now()))
    trainer.save_checkpoint("models/latest.ckpt")

def getModel(new = True, path = None):
    hparams = {
        #also good: 5e-4
        'lr': 1e-3,
        'reg': 0,
        'channels': 20
    }
    if new:
        return ValueNet(SIZE, hparams)
    else:
        if path == None:
            path = "models/latest.ckpt"
        return ValueNet.load_from_checkpoint(path, s=SIZE, hparams=hparams)

def getError(model, dataloader):
    model.eval()
    sum_loss = 0
    cnt = 0
    for (idx, batch) in enumerate(dataloader):
        sum_loss += len(batch) * model.validation_step(batch, idx)
        cnt += len(batch)
    return sum_loss/cnt

def trainModel():
    data = readData("data/data.json")
    dataloader = GameData(data, batch_size=256)
    dataloader.prepare_data()
    model = getModel(new=True)
    trainModel(model, dataloader)
    print(getError(model, dataloader.val_dataloader()))

#trainModel()

class NNBot:
    myColor = 1
    otherColor = 2

    def __init__(self, myColor):
        self.myColor = myColor
        self.otherColor = 3-myColor
        #self.model = getModel(new=False, path="models/net2_100k.ckpt")
        self.model = getModel(new=False)
        self.model.eval()

    def nextMove(self, board):
        posMoves = board.movesAvailable()
        positions = []
        for (y, x) in posMoves:
            if board.wouldWin(self.myColor, y, x):
                return (y, x)
            board.move(self.myColor, y, x)
            positions.append((prepareData(board.board, self.otherColor), (y,x)))
            board.move(0, y, x)
        tensor_data = [(torch.FloatTensor(x).float(), y) for (x,y) in positions]
        dataloader = DataLoader(tensor_data, batch_size=256)
        bestProb = -2.0
        bestMove = (-1, -1)
        for (idx, batch) in enumerate(dataloader):
            probs = self.model(batch[0])
            for i in range(len(batch[1][0])):
                #print(batch[1][0][i], batch[1][1][i], probs[i])
                if probs[i] > bestProb:
                    bestProb = probs[i]
                    bestMove = (batch[1][0][i], batch[1][1][i])
        #print(posMoves)
        #print(bestProb, bestMove)
        #print("win prob of other:", bestProb)
        return bestMove
