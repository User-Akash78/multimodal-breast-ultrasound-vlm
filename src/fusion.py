import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.visual_to_text = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.text_to_visual = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.visual_norm = nn.LayerNorm(
            embed_dim
        )

        self.clinical_norm = nn.LayerNorm(
            embed_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        visual_tokens,
        clinical_tokens,
        clinical_padding_mask=None
    ):

        # Visual features attend to clinical data
        visual_attended, _ = (
            self.visual_to_text(
                query=visual_tokens,
                key=clinical_tokens,
                value=clinical_tokens,
                key_padding_mask=
                    clinical_padding_mask
            )
        )

        visual_fused = self.visual_norm(
            visual_tokens
            + self.dropout(
                visual_attended
            )
        )

        # Clinical features attend to visual data
        clinical_attended, _ = (
            self.text_to_visual(
                query=clinical_tokens,
                key=visual_tokens,
                value=visual_tokens
            )
        )

        clinical_fused = (
            self.clinical_norm(
                clinical_tokens
                + self.dropout(
                    clinical_attended
                )
            )
        )

        return (
            visual_fused,
            clinical_fused
        )