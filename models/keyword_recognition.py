import torch
import torch.nn as nn
import torchaudio.transforms as transforms


class Keyword(nn.Module):
    def __init__(self, n_categories, n_mels=80):
        super().__init__()
        self.n_mels = n_mels
        self.transforms = nn.Sequential(transforms.Resample(orig_freq=16000, new_freq=8000),
                                        transforms.MelSpectrogram(n_mels=self.n_mels, sample_rate=8000),
                                        transforms.AmplitudeToDB())

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, (5, 1))
        self.norm1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 1, (5, 1))
        self.norm2 = nn.BatchNorm2d(1)

        self.blstm = nn.LSTM(n_mels, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 128)
        self.soft_max = nn.Softmax(dim=1)

        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, n_categories)

        self.tsfm = lambda q: q[:, q.shape[1] // 2]

    def forward(self, x):
        # transform
        x = self.transforms(x)
        x = x.permute(0, 1, 3, 2)

        # extraction of local relations
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm2(x)

        # long term dependencies
        x = x.squeeze()  # [batch_size(100), channel(1), spec_len(time, 41), n_mels(80)
        x, _ = self.blstm(x)

        x_mid = self.tsfm(x)
        query = self.fc(x_mid)
        query = query.unsqueeze(-1)
        att_score = torch.bmm(x, query)
        att_score = self.soft_max(att_score)
        x = x.permute(0, 2, 1)
        att_vector = torch.bmm(x, att_score)
        att_vector = att_vector.squeeze()

        output = self.dense1(att_vector)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.soft_max(output)

        return output
