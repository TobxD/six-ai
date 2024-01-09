from collections import defaultdict
import os, copy
import sys, random
import logging
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

logger = logging.getLogger(__file__)

CPU_DEVICE = torch.device("cpu")

def getMCTSBot(player: DictConfig, cfg: DictConfig, color, network=None, randomUpToMove=False):
    bot = MCTSPolicyValueBot(model_path=player.model_path, cfg=cfg, network_conf=player.network_conf, network=network, myColor=color, randomUpToMove=randomUpToMove, numIterations=player.numIterations, c_puct=player.c_puct, dirichletNoise=player.dirichletNoise, randomTemp=player.randomTemp) #MCTSPolicyValueBot(2, network)
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
        self.V = self.V.item()

    def _getPV(self, s: Board, network):
        with profiler.getProfiler("getPV"):
            with profiler.getProfiler("prepare input data"):
                inputData = PVData.prepareInput(s.board, s.toMove)
            with profiler.getProfiler("create input tensor"):
                # X = torch.stack(inputData, 0).to(self.device).unsqueeze(0)
                # X = torch.stack(inputData, 0).to(torch.device("cuda")).unsqueeze(0)
                X = torch.stack(inputData, 0).to(torch.device("cpu")).unsqueeze(0)
            with profiler.getProfiler("eval network"):
                with torch.no_grad():
                    policy, value = network(X)
            with profiler.getProfiler("moves avail"):
                movesAvailable = torch.tensor(s.movesAvailableAsTensor())
            with profiler.getProfiler("prep policy for softmax"):
                policy = policy[0, 0].to(CPU_DEVICE)
                policy[~movesAvailable] = float("-inf")
                policy = torch.softmax(policy.view(-1), 0).view_as(policy)
            P = policy.detach().numpy()[movesAvailable]

        # TODO verify P does the right thing. verify order is as in sorted moves available
        return P, value[0].cpu()

class MCTSPolicyValueBot:
    myColor = 1
    otherColor = 2
    numIterations: int
    logPV: bool

    def __init__(self, myColor, model_path, cfg, network_conf, randomUpToMove = 0, network = None, numIterations = 200, c_puct = 1, dirichletNoise = False, randomTemp=1):
        self.myColor = myColor
        self.randomUpToMove = randomUpToMove
        self.otherColor = 3-myColor
        # self.device = torch.device("cuda")
        self.device = torch.device("cpu")
        self.network = PVnet.getModel(network_conf, cfg.train, model_path)
        self.network.to(self.device)
        self.network.eval()
        self.numIterations = numIterations
        self.c_puct = c_puct
        self.logPV = cfg.play.log_pv
        self.dirichletNoise = dirichletNoise
        self.randomTemp = randomTemp
        self.tree_node_played = None
        self.other_player = None

    def search(self, s: Board, node: Node):
        if node.game_result != None:
            return -node.game_result

        not_visited = node.N == 0
        avg_q = 0 if not_visited.all() else np.sum(node.Q * node.N) / np.sum(node.N)
        node.Q[not_visited] = avg_q
        ucts = node.Q + self.c_puct*node.P*math.sqrt(1+sum(node.N))/(1+node.N)
        a_ind = np.argmax(ucts)
        # do non-visited ones first
        # a_ind = np.argmax(ucts + 1000*not_visited)
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
        is_root = hasattr(node, "dirichlet")
        # docu at: https://docs.python.org/3/library/string.html#formatstrings
        TGREEN =  '\033[32m'
        ENDC = '\033[m'

        log_strs = []
        log_strs.append(f"V: {node.V: 1.4f}")
        log_strs.append(f"worker: {os.getpid()}")
        log_strs.append(str(board))
        log_strs.append("{:^5} {:^5} {:^7} {:^7} {:^7}".format("move", "N", "Q", "P", "V") + (" dirichlet" if is_root else ''))
        for i, move in enumerate(node.moves):
            node_val = "-------" if node.children[i] is None else f"{node.children[i].V: 1.4f}"
            log_strs.append((TGREEN if move==bestMove else '') + 
                            "{:^5} {:^5} {: 1.4f} {: 1.4f} {}".format('-'.join(str(x) for x in move), node.N[i], node.Q[i], float(node.P[i]), node_val) +
                            (" {: 1.4f}".format(node.dirichlet[i]) if is_root else '') +
                            (ENDC if move==bestMove else ''))
        log_strs.append(str(bestMove))
        print("\n".join(log_strs) + "\n")


    def _add_dirichlet_noise(self, node, epsilon=0.25, alpha=1/3):
        node.dirichlet = np.random.dirichlet([alpha for i in range(len(node.children))])
        node.P = (1-epsilon) * node.P + epsilon * node.dirichlet

    def nextMove(self, board):
        if self.other_player and self.other_player.tree_node_played:
            root = self.other_player.tree_node_played
        else:
            root = Node(board, None, None, self.network)
        if self.dirichletNoise:
            self._add_dirichlet_noise(root)

        # take at least 30 iterations to explore using new dirichlet noise
        min_iterations = min(30, self.numIterations) if self.dirichletNoise else 0
        iterations_needed = max(int(self.numIterations - root.N.sum()), min_iterations)
        for _ in range(iterations_needed):
            self.search(board, root)

        tempN = root.N**(1/self.randomTemp)
        policy_arr = tempN / sum(tempN)
        policy = {move:policy_arr[i] for i, move in enumerate(root.moves)}

        if board.numberOfMovesPlayed() < self.randomUpToMove:
            bestMoveInd = np.random.choice(len(root.N), p=policy_arr)
            bestMove = root.moves[bestMoveInd]
        else:
            ### Best move according to N
            # TODO: break ties randomly
            bestMoveInd = np.argmax(root.N)
            bestMove = root.moves[bestMoveInd]

        if self.logPV:
            self.printMCTS(board, root, bestMove)

        self.tree_node_played = root.children[bestMoveInd]

        return bestMove, policy
