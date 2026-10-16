import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class EfficientNetEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()

        backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        self.features = backbone.features

        self.projection = nn.Linear(
            1280,
            output_dim
        )

    def forward(self, image, lesion_mask=None):

        feature_map = self.features(image)

        if lesion_mask is not None:
            small_mask = F.interpolate(
                lesion_mask,
                size=feature_map.shape[-2:],
                mode="nearest"
            )

            feature_map = feature_map * (
                1.0 + small_mask
            )

        tokens = feature_map.flatten(2)
        tokens = tokens.transpose(1, 2)

        tokens = self.projection(tokens)

        return tokens