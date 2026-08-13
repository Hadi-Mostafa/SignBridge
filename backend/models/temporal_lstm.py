"""Word-level ASL sequence classifier for 21 MediaPipe landmarks × 3 coordinates."""
import torch
from torch import nn

class LandmarkLSTM(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 192, layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(63, hidden_size, num_layers=layers, batch_first=True, dropout=.25 if layers > 1 else 0, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Dropout(.3), nn.Linear(hidden_size * 2, num_classes))
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence)[0][:, -1])
