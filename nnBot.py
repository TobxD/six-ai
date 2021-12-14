from math import sqrt, log
from pathlib import Path
import json, random
from datetime import datetime
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pprint

from net import ValueNet, GameData
from util import *

def prepareInput(board, toMove):
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
            data.append((prepareInput(board, toMove), float(result) if toMove == 1 else float(-result)))
    pprint.pprint(data[0])
    return data

def trainModel(model, dataloader):
    trainer = pl.Trainer(
        weights_summary=None,
        max_epochs=50,
        progress_bar_refresh_rate=25,
        gpus=1
    )
    trainer.fit(model, train_dataloaders=dataloader.train_dataloader(), val_dataloaders=dataloader.val_dataloader())
    trainer.save_checkpoint(Path("models/net_{date}.ckpt".format(date=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
    trainer.save_checkpoint(Path("models/latest.ckpt"))

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
            path = Path("models/latest.ckpt")
        return ValueNet.load_from_checkpoint(path, s=SIZE, hparams=hparams)

def getError(model, dataloader):
    model.eval()
    sum_loss = 0
    cnt = 0
    for (idx, batch) in enumerate(dataloader):
        sum_loss += len(batch) * model.validation_step(batch, idx)
        cnt += len(batch)
    return sum_loss/cnt

def trainModelFromData():
    data = readData(Path("data/data.json"))
    dataloader = GameData(data, batch_size=1024)
    dataloader.prepare_data()
    model = getModel(new=True)
    trainModel(model, dataloader)
    print(getError(model, dataloader.val_dataloader()))

#trainModelFromData()

def evalNet(model, board, toMove):
    positions = [(prepareInput(board.board, toMove), 0)]
    tensor_data = [(torch.FloatTensor(x).float(), y) for (x,y) in positions]
    dataloader = DataLoader(tensor_data, batch_size=256)
    for (idx, batch) in enumerate(dataloader):
        return model(batch[0])

class NNBot:
    myColor = 1
    otherColor = 2

    def __init__(self, myColor):
        self.myColor = myColor
        self.otherColor = 3-myColor
        self.model = getModel(new=False, path="models/latest.ckpt")
        #self.model = getModel(new=False)
        self.model.eval()

    def nextMove(self, board):
        posMoves = board.movesAvailable()
        positions = []
        for (y, x) in posMoves:
            if board.wouldWin(self.myColor, y, x):
                return (y, x)
            board.move(self.myColor, y, x)
            positions.append((prepareInput(board.board, self.otherColor), (y,x)))
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
        return bestMove, None

class MCTSValueBot:
    myColor = 1
    otherColor = 2

    def __init__(self, myColor, numIterations):
        self.myColor = myColor
        self.otherColor = 3-myColor
        self.model = getModel(new=False, path="models/net2_100k.ckpt")
        #self.model = getModel(new=False)
        self.model.eval()
        self.numIterations = numIterations
        self.numIterations = 1000
        self.exploration_constant = sqrt(2)

    # watch out, changed q-value to be from view of person who is to move at that node
    # because this eliminates case distinctions
    def newNode(self, board, toMove):
        ind = len(self.searchTree)
        posMoves = board.movesAvailable()
        #TODO uncomment
        #random.shuffle(posMoves)
        winner = board.hasWon()
        if winner != 0:
            val = 1 if toMove == winner else 0
        elif len(board.movesAvailable()) == 0:
            val = 1/2
        else:
            val = evalNet(self.model, board, toMove)
            # 1 would indicate that the player to move wins, 0 that she loses
            val = (1-val)/2
        self.searchTree.append({
            'q': val,
            'n': 1,
            'vis': [],
            'unvis': posMoves
        })
        # 1-val to match view of caller
        return 1-val, ind

    def selectNext(self, curNode):
        bestInd, bestVal = 0, 0
        for i in range(len(curNode['vis'])):
            child = self.searchTree[curNode['vis'][i][1]]
            uct = (1-child['q']/child['n']) + self.exploration_constant * sqrt(log(curNode['n']-1)/child['n'])
            if uct > bestVal:
                bestInd, bestVal = i, uct
        return curNode['vis'][bestInd]

    #toMove: 1 or 2
    def expandLeaf(self, board, curNodeIdx, toMove):
        curNode = self.searchTree[curNodeIdx]
        if board.hasWon() == 0 and len(board.movesAvailable()) != 0:
            if len(curNode['unvis']) != 0:
                posMoves = board.movesAvailable()
                y, x = curNode['unvis'][-1]
                curNode['unvis'] = curNode['unvis'][:-1]
                board.move(toMove, y, x)
                val, ind = self.newNode(board, 3-toMove)
                curNode['vis'].append(((y,x), ind))
                board.move(0, y, x)
            else:
                (y, x), nextNode = self.selectNext(curNode)
                board.move(toMove, y, x)
                val = self.expandLeaf(board, nextNode, 3-toMove)
                board.move(0, y, x)
            curNode['q'] += val
            curNode['n'] += 1
            return 1-val
        else:
            val = curNode['q']/curNode['n']
            curNode['n'] += 1
            curNode['q'] *= val * (curNode['n']+1)
            return 1-val

    def nextMove(self, board):
        self.searchTree = []
        self.newNode(board, self.myColor)
        for i in range(self.numIterations):
            self.expandLeaf(board, 0, self.myColor)
        #pprint.pprint(self.searchTree)
        moveInd = random.randint(1, self.searchTree[0]['n']-1)
        sum = 0
        # expects that we have visited visited all moves once, so self.numIterations should be >= max number of moves in a position
        for (move, ind) in self.searchTree[0]['vis']:
            sum += self.searchTree[ind]['n']
            if sum >= moveInd:
                return move
        # should not happen
        print("error in mcts")
        exit(1)