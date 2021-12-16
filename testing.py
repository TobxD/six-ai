import board

def testWinDetection():
    new_shapes = [[(y+5, x+5) for (y,x) in shape] for shape in board.shapes]
    for shape in new_shapes:
        b = board.Board(15)
        for (y,x) in shape:
            b.move(1, y, x)
        print(b.hasWon())
        assert(b.hasWon() == 1)
        print(b)
        print(b.movesAvailable())

def testWouldWin():
    new_shapes = [[(y+5, x+5) for (y,x) in shape] for shape in board.shapes]
    for shape in new_shapes:
        for i in range(len(shape)):
            b = board.Board(15)
            for j in range(len(shape)-1):
                y, x = shape[(i+j+1)%len(shape)]
                b.move(1, y, x)
            y, x = shape[i]
            #print(b)
            assert(b.wouldWin(1, y, x))

if __name__ == "__main__":
    testWinDetection()
    testWouldWin()