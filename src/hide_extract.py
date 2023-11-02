"""
Low-level module for hiding/extracting bits with prompt token ids to generate stego-text.
"""
import torch
from bitstring import BitStream, ConstBitStream

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
    model,
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
    assert method in ["egs"]
    assert prompt_input_ids.dim() == 2, "prompt_input_ids must be (1, seq_len)."
    assert prompt_input_ids.size(0) == 1, "Only support batch size 1."

    match method:
        case "egs":
            return hide_bits_with_prompt_ids_by_egs(model, prompt_input_ids, bits, **kwargs)
        case _:
            raise ValueError(f"Unknown method: {method}")


def hide_bits_with_prompt_ids_by_egs(
    model,
    prompt_input_ids: torch.Tensor,
    bits: ConstBitStream,
    capacity: float = 0.6,
    threshold: float = 5e-3,
    max_bits_len: int = 5,
    **kwargs,
) -> torch.Tensor:
    """Implementation of enhanced greedy search.

    Args:
        model (PreTrainedModel): Model.
        prompt_input_ids (torch.Tensor): Prompt token ids.
        bits (ConstBitStream): Bitstream.
        capacity (float, optional): Capacity of the prompt. Defaults to 0.6.
        threshold (float, optional): Threshold of the prompt. Defaults to 5e-3.
        max_bits_len (int, optional): Maximum length of the prompt. Defaults to 5.
    """
    bs = BitStream(bits)
    cur_ids = prompt_input_ids

    while bs.pos < len(bs):
        egs_result = search.enhanced_greedy_search(
            model,
            cur_ids,
            capacity=capacity,
            threshold=threshold,
            max_bits_len=max_bits_len,
        )
        new_ids, used_bits = egs_result["comb_ids"], egs_result["trunc_bits"]
        used_bits: int = used_bits.item()
        if used_bits == 0:
            assert new_ids.size(0) == 1
            cur_ids = new_ids
            continue

        # TODO: abort if too long
        actual_bits = min(used_bits, len(bs) - bs.pos)
        tmp_bs: ConstBitStream = bs.read(f"bits:{actual_bits}")
        tmp_bs = ConstBitStream(reversed(tmp_bs))
        cur_idx: int = tmp_bs.read(f"uint:{actual_bits}")
        cur_ids = new_ids[cur_idx].unsqueeze(0)

    return search.enhanced_greedy_search_end(model, cur_ids)


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
    assert method in ["egs"]
    assert prompt_input_ids.dim() == 2, "prompt_input_ids must be (1, seq_len)."
    assert prompt_input_ids.size(0) == 1, "Only support batch size 1."

    match method:
        case "egs":
            return extract_bits_with_prompt_ids_by_egs(model, prompt_input_ids, hide_ids, **kwargs)
        case _:
            raise ValueError(f"Unknown method: {method}")


def extract_bits_with_prompt_ids_by_egs(
    model,
    prompt_input_ids: torch.Tensor,
    hide_ids: torch.Tensor,
    capacity: float = 0.6,
    threshold: float = 5e-3,
    max_bits_len: int = 5,
    **kwargs,
) -> BitStream:
    assert prompt_input_ids.dim() == 2, "prompt_input_ids must be (1, seq_len)."
    assert prompt_input_ids.size(0) == 1, "Only support batch size 1."

    ret_bits = BitStream()
    cur_ids = prompt_input_ids
    while True:
        egs_result = search.enhanced_greedy_search(
            model,
            cur_ids,
            capacity=capacity,
            threshold=threshold,
            max_bits_len=max_bits_len,
        )
        new_ids, used_bits = egs_result["comb_ids"], egs_result["trunc_bits"]
        used_bits: int = used_bits.item()
        if used_bits == 0:
            assert new_ids.size(0) == 1
            cur_ids = new_ids
            continue

        if new_ids.size(1) > hide_ids.size(1):
            break

        for idx, new_id in enumerate(new_ids):
            if (new_id == hide_ids[0, : new_id.size(0)]).all().item():
                # found the target
                tmp_bs = ConstBitStream(uint=idx, length=used_bits)
                tmp_bs = ConstBitStream(reversed(tmp_bs))
                ret_bits += tmp_bs
                cur_ids = new_ids[idx].unsqueeze(0)
                break
        else:
            # cannot find the target
            break

    return ret_bits
