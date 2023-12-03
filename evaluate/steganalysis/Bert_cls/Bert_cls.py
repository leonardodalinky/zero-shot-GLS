import torch.nn as nn
from transformers import AutoModel, BertModel


class BertCls(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.bert_model: BertModel = AutoModel.from_pretrained("prajjwal1/bert-medium")
        self.bert_model.encoder.layer[0:2].requires_grad_(False)
        self.mlp = nn.Linear(512, 2)

    @property
    def device(self):
        return self.bert_model.device

    def forward(self, input_ids, attention_mask):
        x = self.bert_model(input_ids, attention_mask=attention_mask)
        x = x.pooler_output
        x = self.mlp(x)
        return x
