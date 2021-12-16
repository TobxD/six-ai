from collections import defaultdict
import json, os, sys, logging, random
import math
from typing import Dict

sys.path.append('.')
from datetime import datetime
from timeit import default_timer as timer

from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
import torch
from torch.utils import data
from main import simulate, storeGames

from randomBot import RandomBot
import PVData

from board import Board, SIZE
from PVnet import PVnet
from pathlib import Path
import hydra
from hydra.utils import instantiate

from pytorch_lightning import Trainer, seed_everything


class PVBot:
    myColor = 1
    otherColor = 2

    def __init__(self, myColor, model):
        self.myColor = myColor
        self.otherColor = 3-myColor
        if not model:
            model = PVnet.getModel(new=False, path=Path("../../2021-12-13/21-45-46/models/latest.ckpt"))
        self.model = model
        self.model.eval()

    def nextMove(self,board:Board):
        #print (board, self.myColor)
        X = torch.FloatTensor(PVData.prepareInput(board.board,self.myColor))
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
        #print (move2D)
        print (torch.amax(validPolicy).item(), value)
        
        return move2D