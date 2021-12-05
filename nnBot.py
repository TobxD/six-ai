import json

def prepareData(board, toMove):
    ownBoard = [[x==toMove for x in line] for line in board]
    otherBoard = [[x==3-toMove for x in line] for line in board]
    return [ownBoard, otherBoard]

def readData(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]
        for i in range(0, len(lines), 2):
            board, toMove = json.loads(lines[i])
            result = int(lines[i+1])
            data.append((prepareData(board, toMove), result if toMove == 1 else -result))
    print(data[0])
    return data

readData("data.json")