"""Generating prompts for LLMs."""
import contextlib
import csv
import random
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from prompt_template import COVER


@contextlib.contextmanager
def random_state(seed: int):
    # save state
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    yield

    # restore state
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["torch_cuda"])


@contextlib.contextmanager
def gen_prompt_ctx(
    mode: Literal["cover"],
    cover: Path | str | None = None,
    cover_col: str = "plaintext",
):
    """Context manager for generating prompts.

    Args:
        mode (str): Mode.
        cover (Path | str, optional): Cover text file. Defaults to None.
        cover_col (str, optional): Cover column name. Defaults to "plaintext".
    """
    if mode == "cover":
        assert cover is not None, "Cover text is required for 'cover' mode."
        cover = Path(cover)
        assert cover.exists(), f"Cover text file '{cover}' does not exist."
        with cover.open("r") as f:
            reader = csv.DictReader(f)
            assert cover_col in reader.fieldnames, f"Cover column '{cover_col}' not in {cover}."
            cover_text: list[str] = [row[cover_col] for row in reader]
        yield partial(cover_mode_prompt_gen, cover_text=cover_text)
    else:
        raise NotImplementedError(f"Mode '{mode}' is not implemented.")


def cover_mode_prompt_gen(
    n_ctx: int,
    seed: int,
    cover_text: list[str],
    corpus: str = "Unknown",
    **kwargs,
) -> str:
    """Generate prompts for cover mode.

    Args:
        n_ctx (int): Number of context sentences.
        seed (int): Random seed.
        cover_text (list[str]): Cover text.
        corpus (str, optional): Corpus name. Defaults to "Unknown".
    """
    with random_state(seed):
        # choose `n_ctx` context text
        context = random.sample(cover_text, n_ctx)
        context = "\n\n".join(context)
        return COVER.substitute(corpus=corpus, context=context)
