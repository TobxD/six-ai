import random
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

class ValueNet(LightningModule):
    def __init__(self, s, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.model = nn.Sequential(
            #nn.Linear(s*s*2, s*s),
            #nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=self.hparams['channels'], kernel_size=7),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.hparams['channels']*(s-6)*(s-6), 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        pred = self(x)[:,0]
        loss = F.mse_loss(pred, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)[:,0]
        loss = F.mse_loss(pred, y)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        #scheduler = ExponentialLR(optimizer, gamma=0.95)
        #return scheduler
        return optimizer

class GameData(LightningDataModule):
    def __init__(self, games, batch_size=4):
        super().__init__()
        self.games = games
        self.batch_size = batch_size

    def prepare_data(self):
        tensor_data = [(torch.FloatTensor(x).float(), torch.as_tensor(np.array(y)).float()) for (x,y) in self.games]
        train_cnt = min(int(0.8*len(tensor_data))+1, len(tensor_data)-1)
        # don't shuffle, otherwise validation data familiar to net; here done for illustration purposes
        #random.shuffle(tensor_data)
        self.train_data = tensor_data[:train_cnt]
        self.val_data = tensor_data[train_cnt:]
        print(len(self.train_data), len(self.val_data))
        random.shuffle(self.train_data)
        random.shuffle(self.val_data)
        #self.train_data = self.train_data[:256]
        #self.val_data = self.val_data[:256]

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)