import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel


class MultimodalReportGenerator(nn.Module):

    def __init__(
        self,
        model_name="gpt2",
        multimodal_dim=1536
    ):
        super().__init__()

        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.gpt2.config.n_embd
        )

        self.multimodal_projection = nn.Linear(
            multimodal_dim,
            hidden_size
        )

    def forward(
        self,
        multimodal_features,
        input_ids,
        attention_mask=None,
        labels=None
    ):

        token_embeddings = (
            self.gpt2
            .transformer
            .wte(input_ids)
        )

        multimodal_embedding = (
            self.multimodal_projection(
                multimodal_features
            )
            .unsqueeze(1)
        )

        inputs_embeds = torch.cat(
            [
                multimodal_embedding,
                token_embeddings
            ],
            dim=1
        )

        if attention_mask is not None:

            prefix_mask = torch.ones(
                (
                    attention_mask.size(0),
                    1
                ),
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )

            attention_mask = torch.cat(
                [
                    prefix_mask,
                    attention_mask
                ],
                dim=1
            )

        if labels is not None:

            ignore_prefix = torch.full(
                (
                    labels.size(0),
                    1
                ),
                -100,
                device=labels.device,
                dtype=labels.dtype
            )

            labels = torch.cat(
                [
                    ignore_prefix,
                    labels
                ],
                dim=1
            )

        return self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )