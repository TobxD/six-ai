SIZE = 10

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
    [(0,1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)], # circle
]

def moveTo(y, x, dir):
    upd = {}
    upd[RIGHT] = (0,1)
    upd[RIGHT_DOWN] = (1,0)
    upd[LEFT_DOWN] = (1,-1)
    upd[LEFT] = (0,-1)
    upd[LEFT_UP] = (-1, 0)
    upd[RIGHT_UP] = (-1, 1)

    y_dif, x_dif = upd[dir]
    return (y+y_dif, x+x_dif)
