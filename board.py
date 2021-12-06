from util import *

class Board:
    size = 10
    board = [[0]*10 for i in range(10)]

    def __init__(self, size, startPieces=False):
        self.size = size
        self.board = [[0]*size for i in range(size)]
        if startPieces:
            mid = (size-1)//2
            self.board[mid][mid] = 1
            self.board[mid][mid+1] = 2

    def move(self, color, y, x):
        self.board[y][x] = color

    def __getitem__(self, y, x):
        return self.board[y][x]

    def inBounds(self, y, x):
        return y >= 0 and y < self.size and x >= 0 and x < self.size

    def hasLine(self, y, x, dir):
        col = self.board[y][x]
        for i in range(6):
            if not self.inBounds(y, x) or self.board[y][x] != col:
                return False
            y, x = moveTo(y, x, dir)
        return True

    def hasCircle(self, y, x):
        col = self.board[y][x]
        dir = RIGHT
        for i in range(6):
            if not self.inBounds(y, x) or self.board[y][x] != col:
                return False
            y, x = moveTo(y, x, dir)
            dir = (dir+1) % 6
        return True


    def hasTriangle(self, y, x, dir):
        col = self.board[y][x]
        for i in range(6):
            if not self.inBounds(y, x) or self.board[y][x] != col:
                return False
            if i % 2 == 0:
                dir = (dir + 2) % 6
            y, x = moveTo(y, x, dir)
        return True

    def hasWinningShape(self, y, x):
        if not self.inBounds(y, x) or self.board[y][x] == 0:
            return 0
        for dir in range(3):
            if self.hasLine(y, x, dir):
                return self.board[y][x]
        if self.hasCircle(y, x):
            return self.board[y][x]
        for dir in range(2):
            if self.hasTriangle(y, x, dir):
                return self.board[y][x]
        return 0

    def wouldWin(self, color, y, x):
        self.move(color, y, x)
        for shape in shapes:
            for (y_dif, x_dif) in shape:
                winner = self.hasWinningShape(y-y_dif, x-x_dif)
                if winner != 0:
                    self.move(0, y, x)
                    return True
        self.move(0, y, x)
        return False

    def hasWon(self):
        for i in range(self.size):
            for j in range(self.size):
                winner = self.hasWinningShape(i, j)
                if winner != 0:
                    return winner
        return 0

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
