from util import *
import numpy

class Board:
    size = 10
    board = [[0]*10 for i in range(10)]
    toMove = 1
    moves = []
    hashMatrix = []
    hashValue = 0

    def __init__(self, size, startPieces=False):
        self.size = size
        self.board = [[0]*size for i in range(size)]
        if startPieces:
            mid = (size-1)//2
            self.board[mid][mid] = 1
            self.board[mid][mid+1] = 2
        self.hashMatrix = numpy.random.randint(low=0,high=2**31-1,size=(3,size,size)).tolist()
        for y in range(self.size):
            for x in range(self.size):
                self.hashValue ^= self.hashMatrix[self.board[y][x]][y][x]

    def move(self, y, x):
        if self.board[y][x] != 0:
            print ("error!", y,x)
            exit(1)

        self.moves.append((y,x))
        self.hashValue ^= self.hashMatrix[self.board[y][x]][y][x]
        self.board[y][x] = self.toMove
        self.hashValue ^= self.hashMatrix[self.board[y][x]][y][x]
        self._flipMove()

    def undoMove(self):
        y, x = self.moves.pop()
        self.hashValue ^= self.hashMatrix[self.board[y][x]][y][x]
        self.board[y][x] = 0
        self.hashValue ^= self.hashMatrix[self.board[y][x]][y][x]
        self._flipMove()

    def _flipMove(self):
        self.toMove = 3-self.toMove

    def __getitem__(self, y, x):
        return self.board[y][x]
    
    def __hash__(self) -> int:
        return self.hashValue

    def inBounds(self, y, x):
        return y >= 0 and y < self.size and x >= 0 and x < self.size

    def hasSpecificWinningShape(self, y, x, shape):
        if not self.inBounds(y, x) or self.board[y][x] == 0:
            return 0
        color = self.board[y][x]
        for (y_dif, x_dif) in shape:
            new_y, new_x = y+y_dif, x+x_dif
            if not self.inBounds(new_y, new_x) or self.board[y+y_dif][x+x_dif] != color:
                return 0
        return color

    def hasWinningShape(self, y, x):
        for shape in shapes:
            winner = self.hasSpecificWinningShape(y, x, shape)
            if winner != 0:
                return winner
        return 0

    def wouldWin(self, color, y, x):
        self.board[y][x] = color
        for shape in shapes:
            for (y_dif, x_dif) in shape:
                winner = self.hasSpecificWinningShape(y-y_dif, x-x_dif, shape)
                if winner != 0:
                    self.board[y][x] = 0
                    return True
        self.board[y][x] = 0
        return False

    def hasWon(self):
        for i in range(self.size):
            for j in range(self.size):
                winner = self.hasWinningShape(i, j)
                if winner != 0:
                    return winner
        return 0

    def gameResult(self):
        if len(self.movesAvailable()) == 0:
            return 0
        else:
            hasWon = self.hasWon()
            return hasWon*2-3 if hasWon != 0 else None

    def __str__(self):
        res = ""
        for i in range(self.size):
            for x in range(i):
                res += ' '
            for j in range(self.size):
                res += str(self.board[i][j]) + " "
            res += "\n"
        return res

    def movesAvailable(self):
        moves = set()
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0:
                    for dir in range(6):
                        y, x = moveTo(i, j, dir)
                        if self.inBounds(y, x) and self.board[y][x] == 0:
                            moves.add((y,x))
        return list(moves)

    def movesAvailableAsTensor(self):
        moves = self.movesAvailable()
        moveTensor = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for move in moves:
            y,x = move
            moveTensor[y][x] = 1

        return moveTensor

    def convert1Dto2Dindex(self, index: int):
        return (index // self.size , index % self.size)

    def convert2Dto1Dindex(self, index):
        return index[0]*self.size+index[1]
