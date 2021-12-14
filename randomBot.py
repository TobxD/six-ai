import random

from board import Board
from util import *

class RandomBot:
    myColor = 1
    otherColor = 2

    def __init__(self, myColor, search_winning=False, search_losing=False):
        self.myColor = myColor
        self.otherColor = 3-myColor
        self.search_winning = search_winning
        self.search_losing = search_losing

    def nextMove(self, board):
        posMoves = board.movesAvailable()
        if self.search_winning:
            for (y, x) in posMoves:
                if board.wouldWin(self.myColor, y, x):
                    return (y, x), None
        if self.search_losing:
            otherColor = 3-self.myColor
            for (y, x) in posMoves:
                if board.wouldWin(otherColor, y, x):
                    return (y, x), None
        return random.choice(posMoves), None
