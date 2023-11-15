"""
Strategy used in hiding/extracting process for high-quality stego-text.
"""

from abc import ABC, abstractmethod

import torch


class StrategyBase(ABC):
    @property
    @abstractmethod
    def state(self):
        ...

    @abstractmethod
    def update(self, *args, **kwargs):
        ...

    @abstractmethod
    def reset(self):
        ...


class TemperatureAlphaStrategy(StrategyBase):
    def __init__(self, start_temp: float, alpha: float):
        super().__init__()
        self.start_temp = start_temp
        self.alpha = alpha
        # state
        self.cur_temp = start_temp

    @property
    def state(self):
        return {
            "cur_temp": self.cur_temp,
            "start_temp": self.start_temp,
            "alpha": self.alpha,
        }

    def update(self, new_ids_choice_cnt: int | None = None):
        if new_ids_choice_cnt is None or new_ids_choice_cnt > 1:
            self.cur_temp *= self.alpha
        else:
            self.cur_temp = self.start_temp

    def reset(self):
        self.cur_temp = self.start_temp

    @property
    def temperature(self) -> float:
        return self.cur_temp


class LogitsRepeatPenaltyStrategy(StrategyBase):
    def __init__(
        self, penalty: float, delta: float, vocab_size: int, device: torch.device | None = None
    ):
        super().__init__()
        assert penalty >= 0, "Penalty should be non-negative."
        assert delta > 0, "Delta should be positive."
        assert vocab_size > 0, "Vocab size should be positive."
        self.penalty = penalty
        self.delta = delta
        self.vocab_size = vocab_size
        # state
        self.cur_logits_offset = torch.zeros(vocab_size, dtype=torch.double, device=device)

    @property
    def state(self):
        return {
            "cur_logit_offset": self.cur_logits_offset,
            "penalty": self.penalty,
            "delta": self.delta,
            "vocab_size": self.vocab_size,
        }

    def update(self, new_token_id: int):
        assert (
            new_token_id < self.vocab_size
        ), f"Token id {new_token_id} >= Vocab size {self.vocab_size}"
        self.cur_logits_offset += self.delta
        self.cur_logits_offset = self.cur_logits_offset.clamp_max_(0)
        self.cur_logits_offset[new_token_id] -= self.penalty

    def reset(self):
        self.cur_logits_offset = torch.zeros_like(self.cur_logits_offset)

    @property
    def logits_offset(self) -> torch.Tensor:
        return self.cur_logits_offset
