import json, datetime
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

def trainModel(model, dataloader):
    trainer = pl.Trainer(
        weights_summary=None,
        max_epochs=10,
        progress_bar_refresh_rate=25,
        gpus=1
    )
    trainer.fit(model, train_dataloaders=dataloader.train_dataloader(), val_dataloaders=dataloader.val_dataloader())
    trainer.save_checkpoint("models/net_{date:%Y-%m-%d_%H:%M:%S}.ckpt".format(date=datetime.datetime.now()))
    trainer.save_checkpoint("models/latest.ckpt")

def getModel(new = True, path = None):
    hparams = {
        #also good: 5e-4
        'lr': 1e-3,
        'reg': 0,
        'channels': 20
    }
    if new:
        return ValueNet(SIZE, hparams)
    else:
        if path == None:
            path = "models/latest.ckpt"
            return ValueNet.load_from_checkpoint(path, s=SIZE, hparams=hparams)

def getError(model, dataloader):
    sum_loss = 0
    cnt = 0
    for (idx, batch) in enumerate(dataloader):
        sum_loss += len(batch) * model.validation_step(batch, idx)
        cnt += len(batch)
    return sum_loss/cnt

data = readData("data/data.json")
dataloader = GameData(data, batch_size=256)
dataloader.prepare_data()
#model = getModel(new=True)
#trainModel(model, dataloader)
model = getModel(new=False)
print(getError(model, dataloader.val_dataloader()))