import random

from board import Board

class RandomBot:
    myColor = 1
    otherColor = 2

    def __init__(self, myColor, search_winning=False, search_losing=False):
        self.myColor = myColor
        self.otherColor = 3-myColor
        self.search_winning = search_winning
        self.search_losing = search_losing

    def nextMove(self, board: Board):
        posMoves = board.movesAvailable()
        bestMoves = []
        if self.search_winning:
            for (y, x) in posMoves:
                if board.wouldWin(self.myColor, y, x):
                    bestMoves.append((y,x))
        if len(bestMoves) != 0:
            prob = 1/len(bestMoves)
            return random.choice(bestMoves), {move:prob for move in bestMoves}
        if self.search_losing:
            otherColor = 3-self.myColor
            for (y, x) in posMoves:
                if board.wouldWin(otherColor, y, x):
                    bestMoves.append((y,x))
        if len(bestMoves) == 0:
            bestMoves = posMoves
        prob = 1/len(bestMoves)
        return random.choice(bestMoves), {move:prob for move in bestMoves}
