import board
from timing import profiler

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

def testProfiler():
    with profiler.getProfiler("test1"):
        cnt = 0
        for i in range(10**6):
            cnt += 1
    with profiler.getProfiler("test1"):
        with profiler.getProfiler("test2"):
            cnt = 0
            for i in range(10**7):
                cnt += 1
    profiler.printStats()

if __name__ == "__main__":
    #testWinDetection()
    #testWouldWin()
    testProfiler()