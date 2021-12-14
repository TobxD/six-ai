from collections import defaultdict
import json, os, sys, logging, random
import math
from typing import Dict

sys.path.append('../six-ai')
from datetime import datetime
from timeit import default_timer as timer

from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
import torch
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from main import simulate, storeGames

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
            data.append((prepareInput(board, toMove), prepareOutput(move, float(result) if toMove == 2 else float(-result))))
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

class PVBot:
    myColor = 1
    otherColor = 2

    def __init__(self, myColor, network):
        self.myColor = myColor
        self.otherColor = 3-myColor
        self.network = network
        if not network:
            self.model = getModel (new=False, path=Path("../../2021-12-13/21-45-46/models/latest.ckpt"))
        self.model.eval()

    def nextMove(self,board:Board):
        #print (board, self.myColor)
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
        #print (move2D)
        print (torch.amax(validPolicy).item(), value)
        
        return move2D

class MCTSPolicyValueBot:
    myColor = 1
    otherColor = 2
    numIterations: int

    def __init__(self, myColor, network = None, model_path = None, numIterations = 200, c_puct = 1):
        self.myColor = myColor
        self.otherColor = 3-myColor
        self.network = network
        if not network:
            self.network = getModel (new=False, path=Path("../../2021-12-13/22-35-50/models/latest.ckpt"))
        self.network.eval()
        ### TODO: torch.no_grad()
        self.numIterations = numIterations
        self.c_puct = c_puct

    def getPV(self, s: Board):
        X = torch.FloatTensor(prepareInput(s.board,s.toMove)) 
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
            u = self.Q[hash(s)][a] + self.c_puct*self.P[hash(s)][a]*math.sqrt(sum(self.N[hash(s)].values()))/(1+self.N[hash(s)][a])
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

        ### Best move according to N
        bestMove = max(self.N[hash(board)], key=self.N[hash(board)].get, default='')
        
        #print(bestMove)
        return bestMove, self.Q[hash(board)]

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
def testNet(cfg: DictConfig, network = None):
    player2 = instantiate(cfg.mcts_bot, network=network, myColor=2) #MCTSPolicyValueBot(2, network)
    player1 = RandomBot(1, search_winning=True, search_losing=True)

    gameCounter = {-1:0, 0:0, 1:0}
    moves = 0
    start = timer()
    numGames=cfg.bot_test.num_games
    games = []
    for i in range(numGames):
        print(f"Game {i}:")
        board = Board(SIZE, startPieces=True)
        game = simulate(board, player1, player2)
        gameCounter[game[1]] += 1
        moves += 1
        print(f"{gameCounter} in {len(game[0])} moves")

        games.append(game)
        if (cfg.bot_test.break_on_loose and game[1] == -1):
            break

    if cfg.bot_test.store_games:
        storeGames(games, cfg.bot_test.store_path)

    print(i)
    end = timer()
    print(f'Total moves: {moves}, moves per game {moves/i}')
    print(f'elapsed time: {end - start} s')
    print(f'per Game: {(end - start)/numGames} s')
    print(gameCounter)

if __name__ == "__main__":
    #data = readDataWithPolicy(Path("data/data.json"))
    #training()
    #print("Name of the current directory : " + os.path.basename(os.getcwd()))
    #print(sys.path)
    testNet()