from board import Board
from moves import *
from randomBot import RandomBot


def testWinDetection():
    shapes = [
        [(0,0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)], # line 1
        [(0,0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)], # line 2
        [(5,0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5)], # line 3
        [(0,0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)], # triangle 1
        [(0,2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)], # triangle 2
        [(0,1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)], # circle
    ]

    for shape in shapes:
        b = Board(10)
        for (y,x) in shape:
            b.move(1, y, x)
        assert(b.hasWon() == 1)
        print(b)
        print(b.movesAvailable())

# there have to be moves available -> at least one stone already set
def simulate(board, player1, player2, startPlayer = 1):
    players = [player1, player2]
    toMove = startPlayer-1
    while True:
        print(board)
        winner = board.hasWon()
        if winner != 0:
            print(winner, "has won!")
            break
        if len(board.movesAvailable()) == 0:
            print("no more moves possible -> draw")
            break
        move_y, move_x = players[toMove].nextMove(board)
        board.move(toMove+1, move_y, move_x)
        toMove = 1-toMove
        
def testRandom():
    board = Board(10, startPieces=True)
    player1 = RandomBot(1)
    player2 = RandomBot(2)
    simulate(board, player1, player2)

#testWinDetection()
testRandom()
