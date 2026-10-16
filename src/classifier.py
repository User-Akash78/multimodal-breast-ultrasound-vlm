import torch.nn as nn


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1536,
        hidden_dim=512,
        dropout=0.3
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                128
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):
        return self.classifier(x).squeeze(1)