import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F

class Model(LightningModule):
    def __init__(self, s):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(s*s, 1),
            nn.ReLU(),
            nn.Linear(s*s*2, s*s),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = nn.MSELoss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
