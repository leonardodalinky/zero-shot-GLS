"""
Low-level module for hiding/extracting bits with prompt token ids to generate stego-text.
"""
import warnings

import torch
from bitstring import BitStream, ConstBitStream
from transformers import LlamaForCausalLM

import search


######################################
#  __    __  __        __            #
# /  |  /  |/  |      /  |           #
# $$ |  $$ |$$/   ____$$ |  ______   #
# $$ |__$$ |/  | /    $$ | /      \  #
# $$    $$ |$$ |/$$$$$$$ |/$$$$$$  | #
# $$$$$$$$ |$$ |$$ |  $$ |$$    $$ | #
# $$ |  $$ |$$ |$$ \__$$ |$$$$$$$$/  #
# $$ |  $$ |$$ |$$    $$ |$$       | #
# $$/   $$/ $$/  $$$$$$$/  $$$$$$$/  #
#                                    #
######################################
def hide_bits_with_prompt_ids(
    model: LlamaForCausalLM,
    prompt_input_ids: torch.Tensor,
    bits: ConstBitStream,
    method: str,
    **kwargs,
) -> torch.Tensor:
    """Hide bits with prompts as token ids.

    Args:
        model (PreTrainedModel): Model.
        prompt_input_ids (torch.Tensor): Prompt token ids.
        bits (ConstBitStream): Bitstream.
        method (str): Method to hide bits. Currently only support "egs" (enhanced greedy search).
    """
    warnings.warn(
        "This function is deprecated. Use hide_bits_with_prompt_ids_by_egs instead.",
        DeprecationWarning,
    )
    assert method in ["egs"]
    assert prompt_input_ids.dim() == 2, "prompt_input_ids must be (1, seq_len)."
    assert prompt_input_ids.size(0) == 1, "Only support batch size 1."

    match method:
        case "egs":
            return hide_bits_with_prompt_ids_by_egs(model, prompt_input_ids, bits, **kwargs)
        case _:
            raise ValueError(f"Unknown method: {method}")


def hide_bits_with_prompt_ids_by_egs(
    model: LlamaForCausalLM,
    prompt_ids: torch.Tensor,
    bits: ConstBitStream,
    threshold: float = 5e-3,
    max_bpw: int = 5,
    max_new_tokens: int = None,
    complete_sent: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, bool, int]:
    """Implementation of enhanced greedy search.

    Args:
        model (PreTrainedModel): Model.
        prompt_ids (torch.Tensor): Prompt token ids.
        bits (ConstBitStream): Bitstream.
        threshold (float, optional): Threshold of the prompt. Defaults to 5e-3.
        max_bpw (int, optional): Maximum bit length of each token to hide. Defaults to 5.
        max_new_tokens (int, optional): Maximum new tokens to be generated. Defaults to None.
        complete_sent (bool, optional): Whether to complete the sentence. Defaults to False.
    """
    bs = BitStream(bits)
    cur_ids = prompt_ids

    is_truncated = False
    while bs.pos < len(bs):
        egs_result = search.enhanced_greedy_search(
            model,
            cur_ids,
            threshold=threshold,
            max_bits_len=max_bpw,
        )
        new_ids, used_bits = egs_result["comb_ids"], egs_result["trunc_bits"]
        used_bits: int = used_bits.item()
        if used_bits == 0:
            assert new_ids.size(0) == 1
            cur_ids = new_ids
            continue

        actual_bits = min(used_bits, len(bs) - bs.pos)
        tmp_bs: ConstBitStream = bs.read(f"bits:{actual_bits}")
        tmp_bs = ConstBitStream(reversed(tmp_bs))
        cur_idx: int = tmp_bs.read(f"uint:{actual_bits}")
        cur_ids = new_ids[cur_idx].unsqueeze(0)
        if max_new_tokens is not None and cur_ids.size(1) - prompt_ids.size(1) >= max_new_tokens:
            # abort if too long
            is_truncated = True
            break

    if complete_sent:
        return search.enhanced_greedy_search_end(model, cur_ids), is_truncated, bs.pos
    else:
        return cur_ids, is_truncated, len(bs)


#########################################################################
#  ________              __                                     __      #
# /        |            /  |                                   /  |     #
# $$$$$$$$/  __    __  _$$ |_     ______   ______    _______  _$$ |_    #
# $$ |__    /  \  /  |/ $$   |   /      \ /      \  /       |/ $$   |   #
# $$    |   $$  \/$$/ $$$$$$/   /$$$$$$  |$$$$$$  |/$$$$$$$/ $$$$$$/    #
# $$$$$/     $$  $$<    $$ | __ $$ |  $$/ /    $$ |$$ |        $$ | __  #
# $$ |_____  /$$$$  \   $$ |/  |$$ |     /$$$$$$$ |$$ \_____   $$ |/  | #
# $$       |/$$/ $$  |  $$  $$/ $$ |     $$    $$ |$$       |  $$  $$/  #
# $$$$$$$$/ $$/   $$/    $$$$/  $$/       $$$$$$$/  $$$$$$$/    $$$$/   #
#                                                                       #
#########################################################################
def extract_bits_with_prompt_ids(
    model,
    prompt_input_ids: torch.Tensor,
    hide_ids: torch.Tensor,
    method: str,
    **kwargs,
) -> BitStream:
    warnings.warn(
        "This function is deprecated. Use extract_bits_with_prompt_ids_by_egs instead.",
        DeprecationWarning,
    )
    assert method in ["egs"]
    assert prompt_input_ids.dim() == 2, "prompt_input_ids must be (1, seq_len)."
    assert prompt_input_ids.size(0) == 1, "Only support batch size 1."

    match method:
        case "egs":
            return extract_bits_with_prompt_ids_by_egs(model, prompt_input_ids, hide_ids, **kwargs)
        case _:
            raise ValueError(f"Unknown method: {method}")


def extract_bits_with_prompt_ids_by_egs(
    model: LlamaForCausalLM,
    prompt_ids: torch.Tensor,
    hide_ids: torch.Tensor,
    threshold: float = 5e-3,
    max_bpw: int = 5,
    **kwargs,
) -> tuple[BitStream, bool]:
    assert prompt_ids.dim() == 2, "prompt_input_ids must be (1, seq_len)."
    assert prompt_ids.size(0) == 1, "Only support batch size 1."

    ret_bits = BitStream()
    cur_ids = prompt_ids
    is_succeed = True
    while True:
        egs_result = search.enhanced_greedy_search(
            model,
            cur_ids,
            threshold=threshold,
            max_bits_len=max_bpw,
        )
        new_ids, used_bits = egs_result["comb_ids"], egs_result["trunc_bits"]
        used_bits: int = used_bits.item()
        if used_bits == 0:
            assert new_ids.size(0) == 1
            cur_ids = new_ids
            continue

        if new_ids.size(1) > hide_ids.size(1):
            break

        hide_ids_last_token: int = hide_ids[0, new_ids.size(1) - 1].item()
        for idx, new_id in enumerate(new_ids.cpu()):
            new_ids_last_token: int = new_id[-1].item()
            if new_ids_last_token == hide_ids_last_token:
                # found the target
                tmp_bs = ConstBitStream(uint=idx, length=used_bits)
                tmp_bs = ConstBitStream(reversed(tmp_bs))
                ret_bits += tmp_bs
                cur_ids = new_ids[idx].unsqueeze(0)
                break
        else:
            # cannot find the target
            is_succeed = False
            break

    return ret_bits, is_succeed
