"""PyTorch module for generation task using RNN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as tr


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # distilbert
        self.bert = tr.AutoModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # input_ids: (B, S)
        last_hidden_state = self.bert(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, S, D)

        return last_hidden_state

    @property
    def hidden_size(self) -> int:
        return self.bert.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.bert.config.vocab_size

    @property
    def device(self):
        return next(self.parameters()).device


class Generator(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 768, n_layer: int = 2, n_z: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_z = n_z

        self.embedding = nn.Embedding(vocab_size, hidden_size // 2)
        self.lstm = nn.LSTM(
            input_size=hidden_size // 2 + n_z,
            hidden_size=hidden_size,
            num_layers=n_layer,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, z, g_hidden=None):
        # input_ids: (B, S)
        # z: (B, Z)
        x = self.embedding(input_ids)  # (B, S, D)
        B, S, _ = x.size()
        z = torch.cat([z] * S, dim=1).view(
            B, S, self.n_z
        )  # Replicate z inorder to append same z at each time step, (B, S, Z)
        x = torch.cat(
            [x, z], dim=2
        )  # Append z to generator word input at each time step, (B, S, D+Z)

        if g_hidden is not None:
            hidden = g_hidden
        else:
            hidden = (
                torch.zeros((self.n_layer, B, self.hidden_size), device=self.device),
                torch.zeros((self.n_layer, B, self.hidden_size), device=self.device),
            )

        # Get top layer of h_T at each time step and produce logit vector of vocabulary words
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)

        # output: (B, S, V)
        # hidden: ((L, B, D), (L, B, D))
        return output, hidden  # Also return complete (h_T, c_T) incase if we are testing

    @property
    def device(self):
        return self.fc.weight.device


class VAE(nn.Module):
    def __init__(self, n_layer: int = 2, n_z: int = 128):
        super().__init__()
        self.encoder = Encoder()
        self.hidden_to_mu = nn.Linear(self.encoder.hidden_size, n_z)
        self.hidden_to_logvar = nn.Linear(self.encoder.hidden_size, n_z)
        self.generator = Generator(n_layer=n_layer, n_z=n_z)
        self.n_z = n_z

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        G_input: torch.Tensor,
        z: torch.Tensor | None = None,
        G_hidden: list[torch.Tensor] | None = None,
    ):
        # input_ids: (B, S)
        # attention_mask: (B, S)
        # G_input: (B, S)
        # z: (B, Z)
        # G_hidden: ((L, B, D), (L, B, D))

        if z is None:  # If we are testing with z sampled from random noise
            B = input_ids.size(0)
            E_hidden = self.encoder(input_ids, attention_mask)  # Get h_T of Encoder
            mu = self.hidden_to_mu(E_hidden)  # Get mean of lantent z
            logvar = self.hidden_to_logvar(E_hidden)  # Get log variance of latent z
            z = torch.randn([B, self.n_z], device=self.device)  # Noise sampled from ε ~ Normal(0,1)
            z = mu + z * torch.exp(
                0.5 * logvar
            )  # Reparameterization trick: Sample z = μ + ε*σ for backpropogation
            kld = (
                -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, 1).mean()
            )  # Compute KL divergence loss
        else:
            kld = None  # If we are training with given text

        logit, G_hidden = self.generator(G_input, z, G_hidden)
        # logit: (B, S, V)
        # G_hidden: ((L, B, D), (L, B, D))
        # kld: (1,)
        return logit, G_hidden, kld

    @property
    def device(self):
        return self.encoder.device
