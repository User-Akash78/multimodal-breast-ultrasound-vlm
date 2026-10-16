import torch
import torch.nn as nn

from transformers import AutoModel


class ClinicalBERTEncoder(nn.Module):
    def __init__(
        self,
        model_name="emilyalsentzer/Bio_ClinicalBERT"
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(
            model_name
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        return outputs.last_hidden_state