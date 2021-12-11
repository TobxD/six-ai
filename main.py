from board import Board
from util import *
from randomBot import RandomBot
from nnBot import NNBot, MCTSBot
import json, random
from pathlib import Path

def testWinDetection():
    new_shapes = [[(y+5, x+5) for (y,x) in shape] for shape in shapes]
    for shape in new_shapes:
        b = Board(15)
        for (y,x) in shape:
            b.move(1, y, x)
        print(b.hasWon())
        assert(b.hasWon() == 1)
        print(b)
        print(b.movesAvailable())

def testWouldWin():
    new_shapes = [[(y+5, x+5) for (y,x) in shape] for shape in shapes]
    for shape in new_shapes:
        for i in range(len(shape)):
            b = Board(15)
            for j in range(len(shape)-1):
                y, x = shape[(i+j+1)%len(shape)]
                b.move(1, y, x)
            y, x = shape[i]
            #print(b)
            assert(b.wouldWin(1, y, x))

posCnt = {-1:0, 0:0, 1:0}
gameCnt = {-1:0, 0:0, 1:0}

# there have to be moves available -> at least one stone already set
def simulate(board, player1, player2, startPlayer = 1):
    moveNum = 0
    players = [player1, player2]
    toMove = startPlayer-1
    positions = []
    while True:
        #print(board)
        winner = board.hasWon()
        if winner != 0:
            result = winner*2 - 3
            print(winner, "has won after", moveNum, "moves")
            break
        if len(board.movesAvailable()) == 0:
            result = 0
            print("no more moves possible -> draw")
            break
        move = players[toMove].nextMove(board)
        positions.append(json.dumps((board.board, move, toMove+1)))
        move_y, move_x = move
        board.move(toMove+1, move_y, move_x)
        toMove = 1-toMove
        moveNum += 1
    posCnt[result] += moveNum
    gameCnt[result] += 1
    with open(Path("data/data.json"), "a") as f:
        for position in positions[-2:]:
            f.write(position + "\n")
            f.write(json.dumps(result) + "\n")


def testRandom(randomColor = False):
    board = Board(SIZE, startPieces=True)
    if not randomColor or random.choice([True, False]):
        #player1 = MCTSBot(1, numIterations=10)
        player1 = RandomBot(1, search_winning=True, search_losing=True)
        #player1 = NNBot(1)
        player2 = RandomBot(2, search_winning=True, search_losing=True)
    else:
        player1 = RandomBot(1, search_winning=True, search_losing=True)
        player2 = RandomBot(2, search_winning=True, search_losing=True)
        player2 = NNBot(2)
    simulate(board, player1, player2)

def generateGames(cnt, randomColor):
    for i in range(cnt):
        print(i)
        testRandom(randomColor)
        print(posCnt, gameCnt)

#testWinDetection()
#testWouldWin()
generateGames(100000, randomColor=False)