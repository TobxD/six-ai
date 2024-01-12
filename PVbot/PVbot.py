import sys

sys.path.append(".")

import torch

import PVData
from board import Board
from PVnet import PVnet
from hydra.utils import to_absolute_path


class PVBot:
    myColor = 1
    otherColor = 2

    def __init__(self, myColor, model):
        self.myColor = myColor
        self.otherColor = 3 - myColor
        if not model:
            model = PVnet.getModel(
                new=False, path=to_absolute_path("models/latest.ckpt")
            )
        self.model = model
        self.model.eval()

    def nextMove(self, board: Board):
        X = torch.FloatTensor(PVData.prepareInput(board.board, self.myColor))
        policy, value = self.model(X.unsqueeze(0))

        movesAvailable = board.movesAvailable()
        policyForSoftmax = []
        policy = policy.tolist()[0]
        for move in movesAvailable:
            policyForSoftmax.append(policy[board.convert2Dto1Dindex(move)])

        policyForSoftmax = torch.softmax(torch.FloatTensor(policyForSoftmax), 0)

        validPolicy = torch.zeros(board.size * board.size)
        for i, move in enumerate(movesAvailable):
            validPolicy[board.convert2Dto1Dindex(move)] = policyForSoftmax[i]

        move1D = torch.argmax(validPolicy).item()
        move2D = board.convert1Dto2Dindex(move1D)
        print(torch.amax(validPolicy).item(), value)

        return move2D
