import numpy as np

RED = 1
BLACK = 2
RIGHT = 0
RIGHT_DOWN = 1
LEFT_DOWN = 2
LEFT = 3
LEFT_UP = 4
RIGHT_UP = 5

shapes = [
    [(0,0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)], # line 1
    [(0,0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)], # line 2
    [(0,0), (-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5)], # line 3
    [(0,0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)], # triangle 1
    [(-2,2), (-1, 1), (-1, 2), (0, 0), (0, 1), (0, 2)], # triangle 2
    [(0,0), (0, 1), (1, -1), (1, 1), (2, -1), (2, 0)], # circle
]
shapes_numpy = np.array(shapes)

class Board:
    def __init__(self, size, startPieces=False):
        self.size = size
        self.board = [[0]*size for i in range(size)]
        self.stones = [np.zeros((size,size), dtype=bool) for i in range(2)]
        self.toMove = 1
        self.moves = []
        if startPieces:
            mid = (size-1)//2
            self.board[mid][mid] = 1
            self.board[mid][mid+1] = 2
            self.stones[0][mid][mid] = True
            self.stones[1][mid][mid+1] = True

    def numberOfMovesPlayed(self):
        return len(self.moves)

    def move(self, y, x):
        if self.board[y][x] != 0:
            print ("board move error!", y,x)
            exit(1)

        self.moves.append((y,x))
        self.board[y][x] = self.toMove
        self.stones[self.toMove-1][y][x] = True
        self._flipMove()

    def undoMove(self):
        y, x = self.moves.pop()
        self.board[y][x] = 0
        self.stones[2-self.toMove][y][x] = False
        self._flipMove()

    def _flipMove(self):
        self.toMove = 3-self.toMove

    def __getitem__(self, yx_tuple):
        y, x = yx_tuple
        return self.board[y][x]
    
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
        res = 0
        for player in range(2):
            stones = np.argwhere(self.stones[player])
            pattern_positions = stones[:, None, None, :] + shapes_numpy
            valid_mask = ((pattern_positions >= 0) & (pattern_positions < self.size)).all(axis=(-2, -1))
            valid_pattern_positions = pattern_positions[valid_mask]
            pattern_res = self.stones[player][valid_pattern_positions[..., 0], valid_pattern_positions[..., 1]]
            if pattern_res.all(axis=-1).any():
                res = player+1
                break

        return res


    def gameResult(self):
        """
        @return:    -1 if playerId=1 has won, 1 if playerId=2 has won, 0 for draws, None if the game is not terminated
        """
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
        res = self.movesAvailableAsTensor()
        res_pos = np.argwhere(res)
        res_pos = [(y,x) for y,x in res_pos.tolist()]
        return res_pos

    def movesAvailableAsTensor(self):
        all_stones = self.stones[0] | self.stones[1]
        res = np.zeros_like(all_stones)
        res[:, :-1] |= all_stones[:, 1:]
        res[:, 1:] |= all_stones[:, :-1]
        res[:-1, :] |= all_stones[1:, :]
        res[1:, :] |= all_stones[:-1, :]
        res[1:, :-1] |= all_stones[:-1, 1:]
        res[:-1, 1:] |= all_stones[1:, :-1]
        res &= ~all_stones

        return res

    def convert1Dto2Dindex(self, index: int):
        return (index // self.size , index % self.size)

    def convert2Dto1Dindex(self, index):
        return index[0]*self.size+index[1]

    def getNextInDir(self, y, x, dir):
        upd = {}
        upd[RIGHT] = (0,1)
        upd[RIGHT_DOWN] = (1,0)
        upd[LEFT_DOWN] = (1,-1)
        upd[LEFT] = (0,-1)
        upd[LEFT_UP] = (-1, 0)
        upd[RIGHT_UP] = (-1, 1)

        y_dif, x_dif = upd[dir]
        return (y+y_dif, x+x_dif)