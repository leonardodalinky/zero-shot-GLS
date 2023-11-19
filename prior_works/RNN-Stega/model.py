"""PyTorch module for generation task using RNN."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNStega(nn.Module):
    def __init__(
        self,
        vocab_size: int = 384,
        embedding_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, S)
        x = self.embedding(token_ids)  # (B, S, E)
        x, _ = self.rnn(x)  # (B, S, D)
        x = self.fc(x)  # (B, S, V)
        return x

    def forward_train(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward and compute loss.

        Args:
            token_ids: (B, S)

        Returns:
            loss: scalar tensor
        """
        # token_ids: (B, S)
        logits = self.forward(token_ids)  # (B, S, V)
        # shift logits and token_ids
        logits = logits[:, :-1]  # (B, S-1, V)
        token_ids = token_ids[:, 1:]  # (B, S-1)
        # loss
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), token_ids.reshape(-1))
        return loss

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device
