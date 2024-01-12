"""
This code is mainly from 
https://github.com/yukw777/leela-zero-pytorch/blob/master/leela_zero_pytorch/network.py
and modified a bit
"""

import logging
import util
import torchmetrics
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from omegaconf import DictConfig

logger = logging.getLogger(__file__)

DataPoint = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ConvBlock(nn.Module):
    """
    A convolutional block with a convolution layer, batchnorm (with beta) and
    an optional relu
    Note on the bias for the convolutional layer:
    Leela Zero actually uses the bias for the convolutional layer to represent
    the learnable parameters (gamma and beta) of the following batch norm layer.
    This was done so that the format of the weights file, which only has one line
    for the layer weights and another for the bias, didn't have to change when
    batch norm layers were added.
    Currently, Leela Zero only uses the beta term of batch norm, and sets gamma to 1.
    Then, how do you actually use the convolutional bias to produce the same results
    as applying the learnable parameters in batch norm? Let's first take
    a look at the equation for batch norm:
    y = gamma * (x - mean)/sqrt(var - eps) + beta
    Since Leela Zero sets gamma to 1, the equation becomes:
    y = (x - mean)/sqrt(var - eps) + beta
    Now, let `x_conv` be the output of a convolutional layer without the bias.
    Then, we want to add some bias to `x_conv`, so that when you run it through
    batch norm without `beta`, the result is the same as running `x_conv`
    through the batch norm equation with only beta mentioned above. In an equation form:
    (x_conv + bias - mean)/sqrt(var - eps) = (x_conv - mean)/sqrt(var - eps) + beta
    x_conv + bias - mean = x_conv - mean + beta * sqrt(var - eps)
    bias = beta * sqrt(var - eps)
    So if we set the convolutional bias to `beta * sqrt(var - eps)`, we get the desired
    output, and this is what LeelaZero does.
    In Tensorflow, you can tell the batch norm layer to ignore just the gamma term
    by calling `tf.layers.batch_normalization(scale=False)` and be done with it.
    Unfortunately, in PyTorch you can't set batch normalization layers to ignore only
    `gamma`; you can only ignore both `gamma` and `beta` by setting the affine
    parameter to False: `BatchNorm2d(out_channels, affine=False)`. So, ConvBlock sets
    batch normalization to ignore both, then simply adds a tensor after, which
    represents `beta`.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, relu: bool = True
    ):
        super().__init__()
        # we only support odd kernel sizes
        assert kernel_size % 2 == 1

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))  # type: ignore
        self.relu = relu

        # initializations
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x += self.beta.view(1, self.bn.num_features, 1, 1).expand_as(x)
        return F.relu(x, inplace=True) if self.relu else x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, relu=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        return F.relu(out, inplace=True)


class Network(nn.Module):
    def __init__(
        self,
        board_size: int,
        in_channels: int,
        residual_channels: int,
        residual_layers: int,
        legacy: bool = False,
    ):
        super().__init__()
        self.board_size = board_size
        if legacy:
            self.conv_input = ConvBlock(in_channels, residual_channels, 3)
        else:
            self.conv_input = ConvBlock(in_channels, residual_channels, 7)
        self.residual_tower = nn.Sequential(
            *[
                ResBlock(residual_channels, residual_channels)
                for _ in range(residual_layers)
            ]
        )
        self.legacy = legacy
        if legacy:
            self.policy_conv = nn.Sequential(
                ResBlock(residual_channels, residual_channels),
                ConvBlock(residual_channels, 1, 1, relu=False),
            )
        else:
            self.policy_head = nn.Sequential(
                *[ResBlock(residual_channels, residual_channels) for _ in range(3)],
                ConvBlock(residual_channels, 1, 1, relu=False),
            )
        if self.legacy:
            self.value_conv = ConvBlock(residual_channels, 1, 1)
            self.value_fc_1 = nn.Linear(board_size * board_size, 256)
            self.value_fc_2 = nn.Linear(256, 1)
        else:
            self.value_conv = nn.Sequential(
                ResBlock(residual_channels, residual_channels),
                ConvBlock(residual_channels, 256, 1),
            )
            self.value_finish = nn.Sequential(
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(self, planes):
        x = self.conv_input(planes)
        x = self.residual_tower(x)

        if self.legacy:
            pol = self.policy_conv(x)
        else:
            pol = self.policy_head(x).view(
                x.shape[0], 1, self.board_size, self.board_size
            )

        if self.legacy:
            val = self.value_conv(x)
            val = F.relu(self.value_fc_1(torch.flatten(val, start_dim=1)))
            val = torch.tanh(self.value_fc_2(val))
        else:
            val = self.value_conv(x)
            val = torch.max(val, dim=-1).values
            val = torch.max(val, dim=-1).values
            val = torch.tanh(self.value_finish(val))

        return pol, val


class PVnet(Network, pl.LightningModule):  # type: ignore
    def __init__(
        self,
        train_conf: DictConfig,
        # network_conf is passed in as a kwarg by Hydra's instantiate(),
        # so it needs to come last
        network_conf: DictConfig,
    ):
        super().__init__(**network_conf)
        self.save_hyperparameters()  # type: ignore

        self.train_conf = train_conf

        # metrics
        self.train_accuracy = torchmetrics.MeanSquaredError()
        self.val_accuracy = torchmetrics.MeanSquaredError()
        self.test_accuracy = torchmetrics.MeanSquaredError()

    def loss(
        self,
        pred_move: torch.Tensor,
        pred_val: torch.Tensor,
        target_move: torch.Tensor,
        target_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cross_entropy_loss = F.cross_entropy(pred_move, target_move)
        mse_loss = F.mse_loss(pred_val, target_val)
        return mse_loss, cross_entropy_loss, mse_loss + cross_entropy_loss

    def training_step(  # type: ignore
        self, batch: DataPoint, batch_idx: int
    ) -> torch.Tensor:
        planes, target_move, target_val = batch
        pred_move, pred_val = self(planes)
        pred_move = pred_move.view(pred_move.shape[0], -1)
        mse_loss, cross_entropy_loss, loss = self.loss(
            pred_move, pred_val, target_move, target_val
        )
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(
            {
                "train_mse_loss": mse_loss,
                "train_ce_loss": cross_entropy_loss,
                "train_acc": self.train_accuracy(pred_move, target_move),
            }
        )
        return loss

    def validation_step(self, batch: DataPoint, batch_idx: int) -> None:  # type: ignore
        planes, target_move, target_val = batch
        pred_move, pred_val = self(planes)
        pred_move = pred_move.view(pred_move.shape[0], -1)
        mse_loss, cross_entropy_loss, loss = self.loss(
            pred_move, pred_val, target_move, target_val
        )
        self.val_accuracy(pred_move, target_move),
        self.log_dict(
            {
                "val_loss": loss,
                "val_mse_loss": mse_loss,
                "val_ce_loss": cross_entropy_loss,
                "val_acc": self.val_accuracy,
            }
        )

    def test_step(self, batch: DataPoint, batch_idx: int) -> None:  # type: ignore
        planes, target_move, target_val = batch
        pred_move, pred_val = self(planes)
        mse_loss, cross_entropy_loss, loss = self.loss(
            pred_move, pred_val, target_move, target_val
        )
        self.test_accuracy(pred_move, target_move)
        self.log_dict(
            {
                "test_loss": loss,
                "test_mse_loss": mse_loss,
                "test_ce_loss": cross_entropy_loss,
                "test_acc": self.test_accuracy,
            }
        )

    def configure_optimizers(self):
        # taken from leela zero
        # https://github.com/leela-zero/leela-zero/blob/db5569ce8d202f77154f288c21d3f2fa228f9aa3/training/tf/tfprocess.py#L190-L191
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.train_conf.optimizer.lr,
            weight_decay=self.train_conf.optimizer.weight_decay,
        )
        return {
            "optimizer": opt,
        }


def getModel(network_conf, train_cfg, path):
    if path == None:
        print("path empty when getting model")
        return PVnet(train_conf=train_cfg, network_conf=network_conf)
    return PVnet.load_from_checkpoint(
        util.toPath(path),
        s=network_conf.board_size,
        train_conf=train_cfg,
        network_conf=network_conf,
    )
