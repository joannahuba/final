# models/architecture/deepstarr.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSTARRLike(nn.Module):
    def __init__(
        self,
        seq_len,
        conv_channels=(128, 256, 256),
        kernel_sizes=(7, 7, 5),
        dropout=0.4,
        use_padding=False,
        use_logits=False,  
    ):
        super().__init__()

        assert len(conv_channels) == 3
        assert len(kernel_sizes) == 3

        self.use_logits = use_logits

        padding = lambda k: k // 2 if use_padding else 0

        self.conv1 = nn.Conv1d(4, conv_channels[0], kernel_size=kernel_sizes[0], padding=padding(kernel_sizes[0]))
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_sizes[1], padding=padding(kernel_sizes[1]))
        self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_sizes[2], padding=padding(kernel_sizes[2]))

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

        self._to_linear = None
        self._get_conv_output(seq_len)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 256)

        self.classifier = nn.Linear(256, 1)
        self.regressor = nn.Linear(256, 1)

    def _get_conv_output(self, seq_len):
        x = torch.zeros(1, 4, seq_len)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        self._to_linear = x.numel()

    def forward(self, x):
        # x = x.permute(0, 2, 1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        cls_out = self.classifier(x)

        if not self.use_logits:
            cls_out = torch.sigmoid(cls_out)

        reg_out = self.regressor(x)

        return cls_out.squeeze(), reg_out.squeeze()