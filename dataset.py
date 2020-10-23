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
import torchaudio
torchaudio.set_audio_backend('soundfile')
from torchaudio.datasets import SPEECHCOMMANDS


def transform(sample, desired_samples, labels):
    waveform, sample_rate, label, speaker_id, utterance_number = sample
    audio = torch.zeros((1, desired_samples), dtype=torch.float32)
    audio[0,:min(desired_samples, waveform.shape[1])] = waveform[:desired_samples]
    return audio, labels.get(label, 0)


class SpeechCommandsDataset(SPEECHCOMMANDS):
    def __init__(self, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform 

    def __getitem__(self, n):
        item = super().__getitem__(n)

        if self.transform is not None:
            return self.transform(item)
        else:
            return item