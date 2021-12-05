RED = 1
BLACK = 2
RIGHT = 0
RIGHT_DOWN = 1
LEFT_DOWN = 2
LEFT = 3
LEFT_UP = 4
RIGHT_UP = 5

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
