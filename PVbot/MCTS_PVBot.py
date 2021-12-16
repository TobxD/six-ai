from collections import defaultdict
import sys, random
import math
import PVbot.PVnet as PVnet
import PVbot.PVData as PVData

sys.path.append('.')
from datetime import datetime

import torch
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from board import Board, SIZE
from pathlib import Path

def getMCTSBot(cfg: DictConfig, color, network=None, randomMove=False):
    print(cfg)
    color = 2
    network = None
    randomMove = False
    bot = MCTSPolicyValueBot(**cfg.mcts_bot, cfg = cfg, network=network, myColor=color, randomMove=randomMove) #MCTSPolicyValueBot(2, network)
    return bot

class MCTSPolicyValueBot:
    myColor = 1
    otherColor = 2
    numIterations: int

    def __init__(self, myColor, cfg = None, randomMove = False, network = None, model_path = None, numIterations = 200, c_puct = 1):
        self.myColor = myColor
        self.randomMove = randomMove
        self.otherColor = 3-myColor
        self.device = torch.device("cuda")
        #self.device = torch.device("cpu")
        self.network = network
        if not network:
            if not model_path:
                model_path = "../../../models/latest.ckpt"
            self.network = PVnet.getModel(cfg, new=False, path=Path(model_path))
        self.network.to(self.device)
        self.network.eval()
        ### TODO: torch.no_grad()
        self.numIterations = numIterations
        self.c_puct = c_puct

    def getPV(self, s: Board):
        X = torch.FloatTensor(PVData.prepareInput(s.board,s.toMove)).to(self.device)
        policy, value = self.network(X.unsqueeze(0))
        movesAvailable = s.movesAvailable()
        policyForSoftmax = []
        policy = policy.tolist()[0]
        for move in movesAvailable:
            policyForSoftmax.append(policy[s.convert2Dto1Dindex(move)])
        policyForSoftmax = torch.softmax(torch.FloatTensor(policyForSoftmax),0)
        P = {}
        for move, probability in zip(movesAvailable,policyForSoftmax):
            #validPolicy[s.convert2Dto1Dindex(move)] = policyForSoftmax[i]
            P[move] = probability

        return P, value

    def search(self, s: Board):
        gameResult = s.gameResult()
        if gameResult != None:
            if s.toMove == 1:
                gameResult *= -1
            return -gameResult

        if hash(s) not in self.visited:
            self.visited.add(hash(s))
            self.P[hash(s)], v = self.getPV(s)
            return -float(v)
    
        max_u, best_a = -float("inf"), -1
        for a in s.movesAvailable():
            u = self.Q[hash(s)][a] + self.c_puct*self.P[hash(s)][a]*math.sqrt(1+sum(self.N[hash(s)].values()))/(1+self.N[hash(s)][a])
            if u>max_u:
                max_u = u
                best_a = a
        a = best_a
        
        #print (f"Explore a = {a}")
        s.move(*a)
        v = self.search(s)
        s.undoMove()

        
        #print(f"Am Zug: {s.toMove}, a = {a}, v = {v}, Q = {self.Q[hash(s)][a]}, N = {self.N[hash(s)][a]}, hash = {hash(s)}")

        self.Q[hash(s)][a] = (self.N[hash(s)][a] * self.Q[hash(s)][a] + v) / (self.N[hash(s)][a]+1)
        self.N[hash(s)][a] += 1

        #print(f"Am Zug: {s.toMove}, a = {a}, v = {v}, Q = {self.Q[hash(s)][a]}, N = {self.N[hash(s)][a]}, hash = {hash(s)}")
        return -v

    def nextMove(self, board):
        self.N = defaultdict(lambda: defaultdict(lambda: 0))
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.P = {}
        self.visited = set()

        for _ in range(self.numIterations):
            self.search(board)

        #print (self.N[hash(board)])
        #print (self.Q[hash(board)])
        #print (self.P[hash(board)])

        ### Best move according to Q
        #bestval = float("-inf")
        #bestMove = None
        #for move in self.Q[hash(board)]:
        #    if self.N[hash(board)][move] > 0:
        #        if self.Q[hash(board)][move] > bestval:
        #            bestval = self.Q[hash(board)][move]
        #            bestMove = move

        
        #print(bestMove)
        N = self.N[hash(board)]
        sumVal = sum(N.values())
        policy = {key:N[key]/sumVal for key in N}

        if self.randomMove:
            cutoff = random.random()
            bestMove = policy[list(policy.keys())[0]]
            for move in policy:
                cutoff -= policy[move]
                if cutoff <= 0:
                    bestMove = move
                    break
        else:
            ### Best move according to N
            bestMove = max(self.N[hash(board)], key=self.N[hash(board)].get, default='')
        return bestMove, policy