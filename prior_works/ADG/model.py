"""PyTorch module for generation task using RNN."""
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADG(nn.Module):
    def __init__(
        self,
        cell: Literal["lstm", "gru", "rnn"] = "lstm",
        vocab_size: int = 384,
        embedding_dim: int = 384,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.0,
        pad_token_id: int = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        mod = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN,
        }[cell]
        self.rnn = mod(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, input_ids: torch.Tensor, logits: bool = False) -> torch.Tensor:
        # token_ids: (B, S)
        x = self.embedding(input_ids)  # (B, S, E)
        x, _ = self.rnn(x)  # (B, S, D)
        x = self.fc(x)  # (B, S, V)
        if logits:
            # return logits
            return x
        else:
            # return log2-probs
            return torch.log2(F.softmax(x, dim=-1) + 1e-10)

    def forward_train(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward and compute loss.

        Args:
            token_ids: (B, S)

        Returns:
            loss: scalar tensor
        """
        # token_ids: (B, S)
        logits = self.forward(token_ids, logits=True)  # (B, S, V)
        # shift logits and token_ids
        logits = logits[:, :-1]  # (B, S-1, V)
        token_ids = token_ids[:, 1:]  # (B, S-1)
        # loss
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), token_ids.reshape(-1))
        return loss

    def sample(self, x, eos_token_id: int):
        logits = self.forward(x, logits=True)
        prob = F.softmax(logits, dim=-1)[:, -1, :]
        prob[:, eos_token_id] = 0
        prob = prob / prob.sum()
        return torch.multinomial(prob, 1)

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device
