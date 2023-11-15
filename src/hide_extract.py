"""
Low-level module for hiding/extracting bits with prompt token ids to generate stego-text.
"""
import warnings
from typing import Literal, get_args

import torch
from bitstring import BitStream, ConstBitStream
from transformers import LlamaForCausalLM

import codec
import search
from strategy import LogitsRepeatPenaltyStrategy, TemperatureAlphaStrategy

MODE_TYPE = Literal["block", "huffman"]
MODE = get_args(MODE_TYPE)


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
    mode: MODE_TYPE = "block",
    threshold: float = 5e-3,
    temperature: float = 1.0,
    temperature_alpha: float = 1.25,
    max_bpw: int = 5,
    max_new_tokens: int = None,
    complete_sent: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, bool, int]:
    """Implementation of enhanced greedy search.

    Args:
        model (PreTrainedModel): Model.
        prompt_ids (torch.Tensor): Prompt token ids. Shape of (1, seq_len).
        bits (ConstBitStream): Bitstream.
        threshold (float, optional): Threshold of the prompt. Defaults to 5e-3.
        temperature (float, optional): Temperature of the prompt. Defaults to 1.0.
        temperature_alpha (float, optional): Temperature alpha for trivial outputs. Defaults to 1.25.
        max_bpw (int, optional): Maximum bit length of each token to hide. Defaults to 5.
        max_new_tokens (int, optional): Maximum new tokens to be generated. Defaults to None.
        complete_sent (bool, optional): Whether to complete the sentence. Defaults to False.
    """
    bs = BitStream(bits)
    cur_ids = prompt_ids
    temp_strategy = TemperatureAlphaStrategy(temperature, temperature_alpha)
    logits_strategy = LogitsRepeatPenaltyStrategy(
        penalty=4.0,
        delta=0.5,
        vocab_size=model.vocab_size,
        device=model.device,
    )

    is_truncated = False
    while bs.pos < len(bs):
        if max_new_tokens is not None and cur_ids.size(1) - prompt_ids.size(1) >= max_new_tokens:
            # abort if too long
            is_truncated = True
            break

        egs_result = search.enhanced_greedy_search(
            model,
            cur_ids,
            threshold=threshold,
            temperature=temp_strategy.temperature,
            max_bits_len=max_bpw,
            logits_offset=logits_strategy.logits_offset,
        )
        new_ids, trunc_bits, sorted_probs = (
            egs_result["comb_ids"],
            egs_result["trunc_bits"],
            egs_result["sorted_probs"],
        )
        assert new_ids.size(0) == sorted_probs.size(0)
        trunc_bits: int = trunc_bits.item()
        if new_ids.size(0) == 1:
            cur_ids = new_ids
            temp_strategy.update(new_ids.size(0))
            logits_strategy.update(cur_ids[0, -1].item())
            continue

        if mode == "block":
            # encode in "block" way
            actual_bits = min(trunc_bits, len(bs) - bs.pos)
            tmp_bs: ConstBitStream = bs.read(f"bits:{actual_bits}")
            tmp_bs = ConstBitStream(reversed(tmp_bs))
            cur_idx: int = tmp_bs.read(f"uint:{actual_bits}")
            cur_ids = new_ids[cur_idx].unsqueeze(0)
            temp_strategy.update()
            logits_strategy.update(cur_ids[0, -1].item())
        elif mode == "huffman":
            # encode using another huffman coding
            sorted_probs_list: list[float] = sorted_probs.tolist()
            idx2probs = {idx: prob for idx, prob in enumerate(sorted_probs_list)}
            idx2code = codec.huffman.from_frequencies(idx2probs)
            code2idx = {code: idx for idx, code in idx2code.items()}
            max_code_len = max(len(code) for code in code2idx.keys())
            tmp_bits = ""
            while bs.pos < len(bs):
                tmp_bits += str(bs.read("bin:1"))
                if (tgt_idx := code2idx.get(tmp_bits)) is not None:
                    # find the target
                    cur_ids = new_ids[tgt_idx].unsqueeze(0)
                    temp_strategy.update()
                    logits_strategy.update(cur_ids[0, -1].item())
                    break
            else:
                # did not find
                assert len(tmp_bits) < max_code_len, "Read too many bits. Impossible!"
                # only happen when the bs is not long enough
                while len(tmp_bits) < max_code_len:
                    tmp_bits += "0"
                    if (tgt_idx := code2idx.get(tmp_bits)) is not None:
                        # find the target
                        cur_ids = new_ids[tgt_idx].unsqueeze(0)
                        temp_strategy.update()
                        logits_strategy.update(cur_ids[0, -1].item())
                        break
                else:
                    # Impossible
                    assert False, "Impossible!"
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")

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
    mode: MODE_TYPE = "block",
    threshold: float = 5e-3,
    temperature: float = 1.0,
    temperature_alpha: float = 1.25,
    max_bpw: int = 5,
    **kwargs,
) -> tuple[BitStream, bool]:
    assert prompt_ids.dim() == 2, "prompt_input_ids must be (1, seq_len)."
    assert prompt_ids.size(0) == 1, "Only support batch size 1."

    ret_bits = BitStream()
    cur_ids = prompt_ids
    temp_strategy = TemperatureAlphaStrategy(temperature, temperature_alpha)
    logits_strategy = LogitsRepeatPenaltyStrategy(
        penalty=4.0,
        delta=0.5,
        vocab_size=model.vocab_size,
        device=model.device,
    )
    is_succeed = True
    while True:
        egs_result = search.enhanced_greedy_search(
            model,
            cur_ids,
            threshold=threshold,
            temperature=temp_strategy.temperature,
            max_bits_len=max_bpw,
            logits_offset=logits_strategy.logits_offset,
        )
        new_ids, trunc_bits, sorted_probs = (
            egs_result["comb_ids"],
            egs_result["trunc_bits"],
            egs_result["sorted_probs"],
        )
        assert new_ids.size(0) == sorted_probs.size(0)
        trunc_bits: int = trunc_bits.item()
        if new_ids.size(0) == 1:
            cur_ids = new_ids
            temp_strategy.update(new_ids.size(0))
            logits_strategy.update(cur_ids[0, -1].item())
            continue

        if new_ids.size(1) > hide_ids.size(1):
            break

        # hide_ids_last_token: The token of the current hide_ids
        cur_hide_ids_token: int = hide_ids[0, new_ids.size(1) - 1].item()
        if mode == "block":
            for idx, new_id in enumerate(new_ids.cpu()):
                new_ids_last_token: int = new_id[-1].item()
                if new_ids_last_token == cur_hide_ids_token:
                    # found the target
                    tmp_bs = ConstBitStream(uint=idx, length=trunc_bits)
                    tmp_bs = ConstBitStream(reversed(tmp_bs))
                    ret_bits += tmp_bs
                    cur_ids = new_ids[idx].unsqueeze(0)
                    temp_strategy.update()
                    logits_strategy.update(cur_ids[0, -1].item())
                    break
            else:
                # cannot find the target
                is_succeed = False
                break
        elif mode == "huffman":
            # encode using another huffman coding
            sorted_probs_list: list[float] = sorted_probs.tolist()
            idx2probs = {idx: prob for idx, prob in enumerate(sorted_probs_list)}
            idx2code = codec.huffman.from_frequencies(idx2probs)
            for idx, new_id in enumerate(new_ids.cpu()):
                new_ids_last_token: int = new_id[-1].item()
                if new_ids_last_token == cur_hide_ids_token:
                    # found the target
                    code = idx2code[idx]
                    tmp_bs = ConstBitStream(bin=code)
                    ret_bits += tmp_bs
                    cur_ids = new_ids[idx].unsqueeze(0)
                    temp_strategy.update()
                    logits_strategy.update(cur_ids[0, -1].item())
                    break
            else:
                # cannot find the target
                is_succeed = False
                break
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")

    return ret_bits, is_succeed
