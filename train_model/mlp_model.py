import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MLP(nn.Module):
    def __init__(
            self,
            in_features=2048,
            hidden_dim=256,
            num_hidden_layers=3,
            dropout_rate=0.5,
            out_features=1
    ):
        super(MLP, self).__init__()

        self.fc_layers = nn.ModuleList()

        self.fc_layers.append(nn.Linear(in_features, hidden_dim))
        for _ in range(1, num_hidden_layers):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_output = nn.Linear(hidden_dim, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for i, fc in enumerate(self.fc_layers):
            x = F.relu(fc(x))
            x = self.dropout(x)

        out = self.fc_output(x)
        return out
