import json
import pytorch_lightning as pl
import pprint

from net import ValueNet, GameData
from util import *

def prepareData(board, toMove):
    ownBoard = [[float(x==toMove) for x in line] for line in board]
    otherBoard = [[float(x==3-toMove) for x in line] for line in board]
    return [ownBoard, otherBoard]

def readData(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]
        for i in range(0, len(lines), 2):
            board, toMove = json.loads(lines[i])
            result = int(lines[i+1])
            data.append((prepareData(board, toMove), float(result) if toMove == 1 else float(-result)))
    pprint.pprint(data[0])
    return data

hparams = {
    #also good: 5e-4
    'lr': 1e-3,
    'reg': 0,
    'channels': 20
}
model = ValueNet(SIZE, hparams)
data = readData("data/data.json")
dataloader = GameData(data, batch_size=256)
dataloader.prepare_data()
trainer = pl.Trainer(
    weights_summary=None,
    max_epochs=70,
    progress_bar_refresh_rate=25,
    gpus=1
)
trainer.fit(model, train_dataloaders=dataloader.train_dataloader(), val_dataloaders=dataloader.val_dataloader())