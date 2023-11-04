"""
Low-level module for codec using GPT-2 model to convert between token ids and bits.
"""
import base64
import warnings

import torch
import torch.nn.functional as F
from bitstring import Bits, BitStream, ConstBitStream
from transformers import GPT2LMHeadModel
from zgls_utils import huffman

THRES = 1e-8


class DecodeException(Exception):
    pass


@torch.no_grad()
def encode_token_ids(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    max_bits_len: int = 256,
    add_bos_token: bool = True,
) -> BitStream:
    """Encode token ids of plaintext into a bitstream.

    Args:
        model (GPT2LMHeadModel): GPT2 model.
        input_ids (torch.Tensor): Token ids of plaintext, must begin with <bos>. (1, seq_len)
        max_bits_len (int, optional): Maximum length of the bitstream. Defaults to 256.
    """
    assert input_ids.dim() == 2, "input_ids must be (1, seq_len)."
    assert input_ids.size(0) == 1, "Only support batch size 1."
    assert (
        input_ids.size(1) > 1
    ), "input_ids must have at least 2 tokens. The first should be <bos>."
    if add_bos_token:
        input_ids = torch.cat(
            [
                torch.tensor([[model.config.bos_token_id]], dtype=torch.long, device=model.device),
                input_ids,
            ],
            dim=-1,
        )
    assert input_ids[0, 0] == model.config.bos_token_id, "input_ids must start with <bos> token."
    assert input_ids[0, -1] != model.config.eos_token_id, "input_ids must not end with <eos> token."

    ret = BitStream()

    for seq_idx in range(input_ids.size(1) - 1):
        logits: torch.Tensor = (
            model(input_ids[:, : seq_idx + 1]).logits[0, seq_idx].to(dtype=torch.float64)
        )  # (vocab_size)
        probs = F.softmax(logits, dim=-1)
        probs = F.threshold(probs, THRES, THRES, inplace=True)
        freqs: list[float] = probs.tolist()

        i2f = {i: f for i, f in enumerate(freqs)}
        codec_table = huffman.from_frequencies(i2f)
        hf_str = codec_table[input_ids[0, seq_idx + 1].item()]
        if len(ret) + len(hf_str) > max_bits_len:
            # if we are about to exceed the max bits length, we stop encoding
            return ret

        ret += Bits("0b" + hf_str)

    return ret


@torch.no_grad()
def decode_bitstream(
    model: GPT2LMHeadModel,
    bits: ConstBitStream,
    remove_bos_token: bool = True,
) -> torch.Tensor:
    """Decode a bitstream into token ids of plaintext.

    Args:
        model (GPT2LMHeadModel): GPT2 model.
        bits (ConstBitStream): Bitstream.

    Returns:
        torch.Tensor: Token ids of plaintext, without any special tokens. (1, seq_len)
    """
    assert len(bits) > 0, "bits is empty."

    bs: ConstBitStream = bits.copy()
    cur_ids = torch.tensor(
        [[model.config.bos_token_id]], dtype=torch.long, device=model.device
    )  # (1, seq_len)
    while bs.pos < len(bs):
        logits: torch.Tensor = model(cur_ids).logits[0, -1].to(dtype=torch.float64)  # (vocab_size)
        probs = F.softmax(logits, dim=-1)
        probs = F.threshold(probs, THRES, THRES, inplace=True)
        freqs: list[float] = probs.tolist()

        i2f = {i: f for i, f in enumerate(freqs)}
        codec_table = huffman.from_frequencies(i2f)
        rev_table = {v: k for k, v in codec_table.items()}
        max_bits_len = max(len(v) for v in codec_table.values())

        # read bits until we find a match
        bit_str = ""
        while bs.pos < len(bs) and len(bit_str) < max_bits_len:
            bit_str += bs.read("bin:1")
            if bit_str in rev_table:
                break
        else:
            raise DecodeException(f"Cannot find a match in the codec table. Got {bit_str}.")

        cur_ids = torch.cat(
            [
                cur_ids,
                torch.tensor([[rev_table[bit_str]]], dtype=torch.long, device=model.device),
            ],
            dim=-1,
        )

    if remove_bos_token and cur_ids[0, 0] == model.config.bos_token_id:
        cur_ids = cur_ids[:, 1:]
    return cur_ids.to(device=model.device)


def wrap_bits(
    bits: ConstBitStream,
    size_bits=8,
    ef_rounds=4,
    **kwargs,
) -> ConstBitStream:
    """Wrap bits with size flag and EF encoding.

    Args:
        bits (ConstBitStream): Bitstream.
        size_flag_bits (int, optional): Number of bits to indicate the size of the bitstream.
            Defaults to 8, i.e. maximum of 256 bits.
        ef_rounds (int, optional): Number of EF rounds. Defaults to 4.
    """
    size_bs = ConstBitStream(uint=len(bits), length=size_bits)
    bs = size_bs + bits
    for _ in range(max(ef_rounds, 0)):
        bs = _ef_encode(bs)

    return ConstBitStream(bs)


def unwrap_bits(
    bits: ConstBitStream,
    size_bits=8,
    ef_rounds=4,
    **kwargs,
) -> ConstBitStream:
    """Unwrap bits with size flag and EF decoding.

    Args:
        bits (ConstBitStream): Bitstream.
        size_flag_bits (int, optional): Number of bits to indicate the size of the bitstream.
            Defaults to 8, i.e. maximum of 256 bits.
        ef_rounds (int, optional): Number of EF rounds. Defaults to 4.
    """
    bs = bits
    for _ in range(max(ef_rounds, 0)):
        bs = _ef_decode(bits)

    prefix_len = size_bits
    assert len(bs) >= prefix_len, f"bits length is too short. {len(bits)} < {prefix_len}"
    valid_len = bs.peek(f"uint:{size_bits}")

    return ConstBitStream(bs[prefix_len : prefix_len + valid_len])


def bits2base64(bits: ConstBitStream) -> str:
    """Convert bits to base64 string."""
    return base64.b64encode(bits.bytes).decode("ascii")


def base642bits(base64_str: str) -> ConstBitStream:
    """Convert base64 string to bits."""
    return ConstBitStream(bytes=base64.b64decode(base64_str.encode("ascii")))


def _ef_encode(bits: ConstBitStream) -> ConstBitStream:
    bits_copy = bits.copy()
    bs = BitStream()
    state = False
    for b in bits_copy:
        if b == state:
            bs.append([False])
        else:
            bs.append([True])
            state = not state
    return ConstBitStream(bs)


def _ef_decode(bits: ConstBitStream) -> ConstBitStream:
    bits_copy = bits.copy()
    bs = BitStream()
    state = False
    for b in bits_copy:
        if b:
            state = not state
        bs.append([state])
    return ConstBitStream(bs)
