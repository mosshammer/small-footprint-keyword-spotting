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
from pytorch_lightning import Trainer
from dataset import SpeechCommandsDataset, transform
import click
from argparse import ArgumentParser
from functools import partial
from model import SincWakeNet


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--classes', nargs='+', default=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'])
    args = parser.parse_args()

    classes = {label: i + 1 for i, label in enumerate(args.classes)}
    classes['unknown'] = 0

    dataset = SpeechCommandsDataset(root=args.dataset, transform=partial(transform, desired_samples=16000, labels=classes))

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=368, shuffle=True, num_workers=8)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=8)

    model = SincWakeNet({'classes': classes})
   
    trainer = Trainer.from_argparse_args(args, precision=16, gpus=1)
    trainer.fit(model, train_dataloader, valid_dataloader)