from collections import defaultdict
import sys, random
import numpy as np
import math
from xmlrpc.client import Boolean
import PVbot.PVnet as PVnet
import PVbot.PVData as PVData


sys.path.append('.')

import torch
from hydra.utils import to_absolute_path
from omegaconf.dictconfig import DictConfig

from board import Board
from timing import profiler

CPU_DEVICE = torch.device("cpu")

def getMCTSBot(player: DictConfig, cfg: DictConfig, color, network=None, randomUpToMove=False):
    bot = MCTSPolicyValueBot(model_path=player.model_path, cfg=cfg, network=network, myColor=color, randomUpToMove=randomUpToMove, numIterations=player.numIterations, c_puct=player.c_puct) #MCTSPolicyValueBot(2, network)
    return bot

class Node:
    def __init__(self, board, parent, last_move, network):
        self.parent = parent
        self.last_move = last_move
        self.game_result = board.gameResult()
        if self.game_result != None:
            if board.toMove == 1:
                self.game_result *= -1
            self.V = self.game_result
            return
        self.moves = list(sorted(board.movesAvailable()))
        self.children = [None for _ in range(len(self.moves))]
        self.N = np.zeros(len(self.moves))
        self.Q = np.zeros(len(self.moves))
        self.P, self.V = self._getPV(board, network)

    def _getPV(self, s: Board, network):
        with profiler.getProfiler("getPV"):
            with profiler.getProfiler("prepare input data"):
                inputData = PVData.prepareInput(s.board,s.toMove)
            with profiler.getProfiler("create input tensor"):
                # X = torch.stack(inputData, 0).to(self.device).unsqueeze(0)
                X = torch.stack(inputData, 0).to(torch.device("cpu")).unsqueeze(0)
            with profiler.getProfiler("eval network"):
                policy, value = network(X)
            with profiler.getProfiler("moves avail"):
                movesAvailable = torch.tensor(s.movesAvailableAsTensor())
            with profiler.getProfiler("prep policy for softmax"):
                policy = policy[0, 0].to(CPU_DEVICE)
                policy[~movesAvailable] = float("-inf")
                policy = torch.softmax(policy.view(-1), 0).view_as(policy)
            P = policy.detach().numpy()[movesAvailable]

        # TODO verify P does the right thing. verify order is as in sorted moves available
        return P, value[0]

class MCTSPolicyValueBot:
    myColor = 1
    otherColor = 2
    numIterations: int
    logPV: bool

    def __init__(self, myColor, model_path, cfg, randomUpToMove = 0, network = None, numIterations = 200, c_puct = 1):
        self.myColor = myColor
        self.randomUpToMove = randomUpToMove
        self.otherColor = 3-myColor
        #self.device = torch.device("cuda")
        self.device = torch.device("cpu")
        self.network = PVnet.getModel(cfg, model_path)
        self.network.to(self.device)
        self.network.eval()
        ### TODO: torch.no_grad()
        self.numIterations = numIterations
        self.c_puct = c_puct
        self.logPV = cfg.play.log_pv

    def search(self, s: Board, node: Node):
        if node.game_result != None:
            return -node.game_result

        ucts = node.Q + self.c_puct*node.P*math.sqrt(1+sum(node.N))/(1+node.N)
        a_ind = np.argmax(ucts)
        a = node.moves[a_ind]
        
        s.move(*a)
        if node.children[a_ind] == None:
            node.children[a_ind] = Node(s, node, a, self.network)
            v = -node.children[a_ind].V
        else:
            v = self.search(s, node.children[a_ind])
        s.undoMove()

        node.Q[a_ind] = (node.N[a_ind] * node.Q[a_ind] + v) / (node.N[a_ind]+1)
        node.N[a_ind] += 1

        return -v

    def printMCTS (self, board, node, bestMove):
        # docu at: https://docs.python.org/3/library/string.html#formatstrings
        TGREEN =  '\033[32m'
        ENDC = '\033[m'

        print (board)
        print("{:^5} {:^5} {:^7} {:^7}".format("move", "N", "Q", "P"))
        for i, move in enumerate(node.moves):
            print(  (TGREEN if move==bestMove else '') + 
                    "{:^5} {:^5} {: 1.4f} {: 1.4f}".format('-'.join(str(x) for x in move), node.N[i], node.Q[i], float(node.P[i])) +
                    (ENDC if move==bestMove else ''))
        print (bestMove)

    def _add_dirichlet_noise(self, node, epsilon=0.25, alpha=1/3):
        dirichlet = np.random.dirichlet([alpha for i in range(len(node.children))])
        node.P = (1-epsilon) * node.P + epsilon * dirichlet

    def nextMove(self, board):
        print("moving with mcts")
        root = Node(board, None, None, self.network)
        self._add_dirichlet_noise(root)

        with profiler.getProfiler("mcts iterations"):
            for _ in range(self.numIterations):
                self.search(board, root)
        policy = root.N / sum(root.N)
        policy = {move:policy[i] for i, move in enumerate(root.moves)}

        if board.numberOfMovesPlayed() < self.randomUpToMove:
            cutoff = random.random()
            bestMove = policy[list(policy.keys())[0]]
            for move in policy:
                cutoff -= policy[move]
                if cutoff <= 0:
                    bestMove = move
                    break
        else:
            ### Best move according to N
            # TODO: break ties randomly
            bestMoveInd = np.argmax(root.N)
            bestMove = root.moves[bestMoveInd]

        if self.logPV:
            self.printMCTS(board, root, bestMove)
        return bestMove, policy
