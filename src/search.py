"""
Fundamental search algorithms for LLMs.
"""
import json
import os.path as osp

import torch
import torch.nn.functional as F
import transformers as tr

# Token IDs to ignore during geneartion.
DEFAULT_IGNORED_IDS = set(
    [
        0,
        1,
        2,
        12,  # token `\t`
        13,  # token `\n`
        6756,  # token `\r`
    ]
)
with open(f"{osp.dirname(__file__)}/ignore_ids.json", "r") as fp:
    DEFAULT_IGNORED_IDS = DEFAULT_IGNORED_IDS.union(set(json.load(fp)))
DEFAULT_IGNORED_IDS = list(DEFAULT_IGNORED_IDS)


@torch.no_grad()
def beam_search(
    model,
    prompt_input_ids: torch.Tensor,
    num_beams: int = 4,
    num_return_sequences: int = 4,
    new_tokens: int = 2,
    fallback_eos_id: int = 1126,  # token `And` in LLaMa
) -> torch.Tensor:
    assert prompt_input_ids.dim() == 2
    assert prompt_input_ids.size(0) == 1, "Only support batch size 1."
    model: tr.AutoModelForCausalLM

    generation_config = tr.GenerationConfig.from_model_config(model.config)
    eos_id = generation_config.eos_token_id
    generation_config.eos_token_id = None
    generation_config.length_penalty = 10.0
    generation_config.num_beams = num_beams
    generation_config.num_return_sequences = num_return_sequences
    generation_config.min_new_tokens = new_tokens
    generation_config.max_new_tokens = new_tokens

    ret_ids: torch.Tensor = model.generate(
        prompt_input_ids, generation_config=generation_config
    )  # (num_return_sequences, seq_len)
    # replace every occurence of `eos_id` with `fallback_eos_id`
    ret_ids[ret_ids == eos_id] = fallback_eos_id
    ret_ids[ret_ids == generation_config.pad_token_id] = fallback_eos_id
    ret_ids[ret_ids == 2] = fallback_eos_id
    return ret_ids


@torch.no_grad()
def beam_search_end(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    eos_ids: int = [1, 29889, 29973, 29991],
) -> torch.Tensor:
    assert input_ids.dim() == 2, f"input_ids.dim() = {input_ids.dim()}"
    assert input_ids.size(0) == 1, "Only support batch size 1."
    model: tr.AutoModelForCausalLM

    generation_config = tr.GenerationConfig.from_model_config(model.config)
    # generation_config.length_penalty = -1.0
    # generation_config.num_beams = 1
    generation_config.num_return_sequences = 1
    generation_config.eos_token_id = eos_ids
    # generation_config.early_stopping = True
    generation_config.max_new_tokens = 32

    ret_ids: torch.Tensor = model.generate(
        input_ids, attention_mask=attention_mask, generation_config=generation_config
    )  # (1, seq_len)

    return ret_ids


@torch.no_grad()
def enhanced_greedy_search(
    model: tr.AutoModelForCausalLM,
    input_ids: torch.Tensor,
    ignored_ids: list[int] = DEFAULT_IGNORED_IDS,
    # fallback_eos_id: int = 1126,  # token `And` in LLaMA
    threshold: float = 5e-3,
    temperature: float = 1.0,
    max_bits_len: int = 5,
    logits_offset: torch.Tensor | None = None,  # (vocab_size,)
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: Probably a LLaMa model.
        input_ids: (batch_size, seq_len) tensor of token ids,
            DO NOT add special tokens to the end.
        # fallback_eos_id: The token id to prevent the model from being stopped.
        ignored_ids: The token ids to be ignored.
        threshold: The minimum probability of the search.
        max_bits_len: The maximum length of the bits.
    """
    assert input_ids.dim() == 2
    assert input_ids.size(0) == 1, "Only support batch size 1."

    logits: torch.Tensor = model(input_ids=input_ids).logits[0, -1]  # (vocab_size)
    logits = logits.double()
    logits /= temperature
    logits += logits_offset
    logits[ignored_ids] = -10
    probs = F.softmax(logits, dim=-1)  # (vocab_size)

    sorted_probs, sorted_probs_indice = probs.sort(dim=-1, descending=True, stable=True)

    less_thres_cnt: torch.Tensor = (sorted_probs >= threshold).sum(dtype=torch.long)

    raw_trunc_cnt = less_thres_cnt
    raw_trunc_cnt = torch.maximum(raw_trunc_cnt, torch.ones_like(raw_trunc_cnt))
    raw_trunc_cnt = torch.minimum(
        raw_trunc_cnt, (2 * torch.ones_like(raw_trunc_cnt)) ** max_bits_len
    )

    trunc_bits = torch.floor(torch.log2(raw_trunc_cnt.float())).long()

    ret_ids = torch.cat(
        [
            input_ids.repeat(raw_trunc_cnt, 1),
            sorted_probs_indice[:raw_trunc_cnt, None],
        ],
        dim=-1,
    )

    return {
        "comb_ids": ret_ids,  # (raw_trunc_cnt, seq_len + 1)
        "trunc_bits": trunc_bits,  # (,)
        "sorted_probs": sorted_probs[:raw_trunc_cnt],  # (raw_trunc_cnt,)
    }


@torch.no_grad()
def enhanced_greedy_search_end(
    model: tr.AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    extra_eos_ids: int = [29889, 29973, 29991],
) -> torch.Tensor:
    eos_ids = [model.config.eos_token_id] + extra_eos_ids
    return beam_search_end(model, input_ids, attention_mask=attention_mask, eos_ids=eos_ids)
