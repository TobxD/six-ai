from collections import defaultdict
import sys, random
import math
import PVbot.PVnet as PVnet
import PVbot.PVData as PVData

sys.path.append('.')

import torch
from hydra.utils import to_absolute_path
from omegaconf.dictconfig import DictConfig

from board import Board
from timing import profiler

CPU_DEVICE = torch.device("cpu")

def getMCTSBot(player: DictConfig, cfg: DictConfig, color, network=None, randomMove=False):
    print(cfg)
    bot = MCTSPolicyValueBot(model_path=player.model_path, cfg = cfg, network=network, myColor=color, randomMove=randomMove, numIterations=player.numIterations, c_puct=player.c_puct) #MCTSPolicyValueBot(2, network)
    return bot

class MCTSPolicyValueBot:
    myColor = 1
    otherColor = 2
    numIterations: int

    def __init__(self, myColor, model_path, cfg = None, randomMove = False, network = None, numIterations = 200, c_puct = 1):
        self.myColor = myColor
        self.randomMove = randomMove
        self.otherColor = 3-myColor
        #self.device = torch.device("cuda")
        self.device = torch.device("cpu")
        self.network = PVnet.getModel(cfg, model_path)
        self.network.to(self.device)
        self.network.eval()
        ### TODO: torch.no_grad()
        self.numIterations = numIterations
        self.c_puct = c_puct

    def getPV(self, s: Board):
        with profiler.getProfiler("getPV"):
            with profiler.getProfiler("prepare input data"):
                inputData = PVData.prepareInput(s.board,s.toMove)
            with profiler.getProfiler("create input tensor"):
                X = torch.FloatTensor(inputData).to(self.device).unsqueeze(0)
                X = X.repeat(1, 1, 1, 1)
            with profiler.getProfiler("eval network"):
                policy, value = self.network(X)
            with profiler.getProfiler("moves avail"):
                movesAvailable = s.movesAvailable()
            with profiler.getProfiler("prep policy for softmax"):
                policyForSoftmax = []
                policy = policy[0].to(CPU_DEVICE)
                with profiler.getProfiler("iterate over moves avail"):
                    possibleMoves = torch.zeros(s.size**2, dtype=torch.bool, device=CPU_DEVICE)
                    for move in movesAvailable:
                        possibleMoves[s.convert2Dto1Dindex(move)] = True
                        #policyForSoftmax.append(policy[s.convert2Dto1Dindex(move)])
                with profiler.getProfiler("convert policy to torch"):
                    #print(policyForSoftmax)
                    #policyTorch = torch.FloatTensor(policyForSoftmax)
                    policyTorch = policy[possibleMoves]
                with profiler.getProfiler("actual softmax"):
                    policyForSoftmax = torch.softmax(policyTorch,0)
            P = {}
            for move, probability in zip(movesAvailable,policyForSoftmax):
                #validPolicy[s.convert2Dto1Dindex(move)] = policyForSoftmax[i]
                P[move] = probability

        return P, value[0]

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

    def printMCTS (self, board, N: defaultdict, Q, P, bestMove):
        # docu at: https://docs.python.org/3/library/string.html#formatstrings
        TGREEN =  '\033[32m'
        ENDC = '\033[m'

        print (board)
        print("{:^5} {:^5} {:^7} {:^7}".format("move", "N", "Q", "P"))
        for key in sorted(N.keys()):
            print(  (TGREEN if key==bestMove else '') + 
                    "{:^5} {:^5} {: 1.4f} {: 1.4f}".format('-'.join(str(x) for x in key), N[key], Q[key], float(P[key])) +
                    (ENDC if key==bestMove else ''))
        print (bestMove)

    def nextMove(self, board):
        self.N = defaultdict(lambda: defaultdict(lambda: 0))
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.P = {}
        self.visited = set()

        with profiler.getProfiler("mcts iterations"):
            for _ in range(self.numIterations):
                self.search(board)

        ### Best move according to Q
        #bestval = float("-inf")
        #bestMove = None
        #for move in self.Q[hash(board)]:
        #    if self.N[hash(board)][move] > 0:
        #        if self.Q[hash(board)][move] > bestval:
        #            bestval = self.Q[hash(board)][move]
        #            bestMove = move

        
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

        self.printMCTS(board, self.N[hash(board)], self.Q[hash(board)], self.P[hash(board)], bestMove)
        return bestMove, policy

   