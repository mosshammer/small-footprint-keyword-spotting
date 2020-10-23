# Copyright 2020 Alexander Moßhammer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
import pytorch_lightning as pl
from sincconv import SincConv


class SincWakeNet(pl.LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.classes = self.hparams['classes']
        self.sinc = SincConv(in_channels=1, out_channels=40, kernel_size=101, stride=8)
        self.norm = nn.BatchNorm1d(num_features=40)
        self.blocks = nn.Sequential(
            self.make_block(in_channels=40,  out_channels=160, kernel_size=25, stride=2),
            self.make_block(in_channels=160, out_channels=160, kernel_size=9, stride=1),
            self.make_block(in_channels=160, out_channels=160, kernel_size=9, stride=1),
            self.make_block(in_channels=160, out_channels=160, kernel_size=9, stride=1),
            self.make_block(in_channels=160, out_channels=160, kernel_size=9, stride=1),
        )
        self.linear = nn.Linear(in_features=160, out_features=len(self.classes))
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sinc(x)
        x = torch.log(torch.abs(x) + 1)
        x = self.norm(x)
        x = nn.functional.avg_pool1d(x, kernel_size=2)
        x = self.blocks(x)
        return self.linear(x.mean([2]))

    def training_step(self, batch, batch_index: int):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        self.log('training/loss', loss)
        return loss

    def validation_step(self, batch, batch_index: int):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        return {'val_loss': loss, 'y_true': labels, 'y_pred': torch.argmax(logits, dim=1)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y_true = torch.cat([x['y_true'] for x in outputs]).cpu().numpy().flatten()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).cpu().numpy().flatten()
        accuracy = metrics.accuracy_score(y_true, y_pred)
        self.log('validation/loss', avg_loss)
        self.log('validation/accuracy', accuracy)
        return avg_loss

    def make_block(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=in_channels, bias=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(p=0.1),
        )