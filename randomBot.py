import random

from board import Board
from util import *

class RandomBot:
    myColor = 1
    otherColor = 2

    def __init__(self, myColor):
        self.myColor = myColor
        self.otherColor = 3-myColor

    def nextMove(self, board):
        posMoves = board.movesAvailable()
        return random.choice(posMoves)
