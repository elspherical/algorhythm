import torch
import torch.nn as nn

class VideoToEmotion(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=13):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.model(x)