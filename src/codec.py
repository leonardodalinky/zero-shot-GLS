import warnings
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from bitstring import Bits, BitStream, ConstBitStream
from joblib import Parallel, delayed
from zgls_utils import huffman

THRES = 1e-9


class DecodeException(Exception):
    pass


@torch.no_grad()
def encode_token_ids(
    model, input_ids: torch.Tensor, max_bits_len: int = 256
) -> Tuple[BitStream, torch.Tensor]:
    assert input_ids.dim() == 2, "input_ids must be (1, seq_len)."
    assert input_ids.size(0) == 1, "Only support batch size 1."
    assert (
        input_ids.size(1) > 1
    ), "input_ids must have at least 2 tokens. The first should be <bos>."
    if input_ids[0, 0] != model.config.bos_token_id:
        warnings.warn("input_ids does not start with <bos> token.")

    ret = BitStream()

    for seq_idx in range(input_ids.size(1) - 1):
        logits: torch.Tensor = (
            model(input_ids[:, : seq_idx + 1]).logits[0, seq_idx].to(dtype=torch.float64)
        )  # (vocab_size)
        probs = F.softmax(logits, dim=-1)
        probs = F.threshold(probs, THRES, THRES, inplace=True)
        freqs: torch.Tensor = probs / probs.min()  # (vocab_size)
        freqs = freqs.round_().long()  # (vocab_size)
        freq = freqs.tolist()

        i2f = {i: f for i, f in enumerate(freq)}
        codec_table = huffman.from_frequencies(i2f)
        hf_str = codec_table[input_ids[0, seq_idx + 1].item()]
        if len(ret) + len(hf_str) > max_bits_len:
            # if we are about to exceed the max bits length, we stop encoding
            return ret, input_ids[:, : seq_idx + 1]

        ret += Bits("0b" + hf_str)

    return ret, input_ids


@torch.no_grad()
def batch_encode_token_ids(
    model,
    input_idss: List[torch.Tensor],
    max_bits_len: int = 128,
    joblib_ctx: Optional[Parallel] = None,
) -> List[Tuple[BitStream, torch.Tensor]]:
    assert isinstance(input_idss, list)
    for input_ids in input_idss:
        assert input_ids.dim() == 2, "input_ids must be (1, seq_len)."
        assert input_ids.size(0) == 1, "Only support batch size 1."
        assert (
            input_ids.size(1) > 1
        ), "input_ids must have at least 2 tokens. The first should be <bos>."
        if input_ids[0, 0] != model.config.bos_token_id:
            warnings.warn("input_ids does not start with <bos> token.")

    max_seq_len = max(input_ids.size(1) for input_ids in input_idss)

    rets = list((BitStream(), input_ids[:, :1]) for input_ids in input_idss)
    finished = [False] * len(input_idss)

    for seq_idx in range(max_seq_len - 1):
        freqs_list = list()

        for batch_idx, input_ids in enumerate(input_idss):
            if finished[batch_idx] or seq_idx >= input_ids.size(1) - 1:
                finished[batch_idx] = True
                # ignore the last token
                freqs_list.append(None)
                continue

            logits: torch.Tensor = (
                model(input_ids[:, : seq_idx + 1])
                .logits[0, seq_idx]
                .to(dtype=torch.float64, non_blocking=True)
            )  # (vocab_size)
            probs = F.softmax(logits, dim=-1)
            probs = F.threshold(probs, THRES, THRES, inplace=True)
            freqs: torch.Tensor = probs / probs.min()  # (vocab_size)
            freqs = freqs.round_().long()  # (vocab_size)
            freqs_list.append(freqs)

        freq_list: List[List[int]] = list()
        for batch_idx, freqs in enumerate(freqs_list):
            if finished[batch_idx]:
                freq_list.append(None)
                continue
            freq = freqs.tolist()
            freq_list.append(freq)

        # deal with context
        if joblib_ctx is None:
            joblib_ctx = Parallel(n_jobs=-1)

        ctx_to_used = joblib_ctx if not joblib_ctx._managed_backend else nullcontext(joblib_ctx)

        with ctx_to_used as parallel:
            codec_table_list = parallel(delayed(_huffman_table)(freq) for freq in freq_list)

            for batch_idx, input_ids in enumerate(input_idss):
                if finished[batch_idx]:
                    continue

                bs, _ = rets[batch_idx]
                codec_table = codec_table_list[batch_idx]
                hf_str = codec_table[input_ids[0, seq_idx + 1].item()]
                if len(bs) + len(hf_str) > max_bits_len:
                    # if we are about to exceed the max bits length, we stop encoding
                    finished[batch_idx] = True

                if not finished[batch_idx]:
                    bs += Bits("0b" + hf_str)
                    rets[batch_idx] = (bs, input_idss[batch_idx][:, : seq_idx + 2])

    return rets


@torch.no_grad()
def decode_bitstream(model, bits: ConstBitStream) -> torch.Tensor:
    assert len(bits) > 0, "bits is empty."

    # bs = bits
    bs: ConstBitStream = bits.copy()
    cur_ids = torch.tensor(
        [[model.config.bos_token_id]], dtype=torch.long, device=model.device
    )  # (1, seq_len)
    while bs.pos < len(bs):
        logits: torch.Tensor = model(cur_ids).logits[0, -1].to(dtype=torch.float64)  # (vocab_size)
        probs = F.softmax(logits, dim=-1)
        probs = F.threshold(probs, THRES, THRES, inplace=True)
        freqs: torch.Tensor = probs / probs.min()  # (1, seq_len, vocab_size)
        freqs = freqs.round_().long()  # (vocab_size)
        freq = freqs.tolist()

        i2f = {i: f for i, f in enumerate(freq)}
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

    return cur_ids.to(device=model.device)


@torch.no_grad()
def batch_decode_bitstream(
    model, bitss: List[ConstBitStream], joblib_ctx: Optional[Parallel] = None
) -> List[torch.Tensor]:
    assert isinstance(bitss, list)
    for idx, bits in enumerate(bitss):
        assert len(bits) > 0, f"{idx}-th bits is empty."

    rets = list(
        torch.tensor([[model.config.bos_token_id]], dtype=torch.long, device=model.device)
        for _ in bitss
    )
    finished = [False] * len(bitss)
    bss = [bits.copy() for bits in bitss]

    while not all(finished):
        freqs_list = list()
        for batch_idx, cur_ids in enumerate(rets):
            if finished[batch_idx]:
                freqs_list.append(None)
                continue
            logits: torch.Tensor = (
                model(cur_ids).logits[0, -1].to(dtype=torch.float64)
            )  # (vocab_size)
            probs = F.softmax(logits, dim=-1)
            probs = F.threshold(probs, THRES, THRES, inplace=True)
            freqs: torch.Tensor = probs / probs.min()  # (1, seq_len, vocab_size)
            freqs = freqs.round_().long()  # (vocab_size)
            freqs_list.append(freqs)

        freq_list: List[List[int]] = list()
        for batch_idx, freqs in enumerate(freqs_list):
            if finished[batch_idx]:
                freq_list.append(None)
                continue
            freq = freqs.tolist()
            freq_list.append(freq)

        # deal with context
        if joblib_ctx is None:
            joblib_ctx = Parallel(n_jobs=-1)

        ctx_to_used = joblib_ctx if not joblib_ctx._managed_backend else nullcontext(joblib_ctx)

        with ctx_to_used as parallel:
            codec_table_list = parallel(delayed(_huffman_table)(freq) for freq in freq_list)

            for batch_idx, bs in enumerate(bss):
                if finished[batch_idx]:
                    continue

                codec_table = codec_table_list[batch_idx]
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

                rets[batch_idx] = torch.cat(
                    [
                        rets[batch_idx],
                        torch.tensor([[rev_table[bit_str]]], dtype=torch.long, device=model.device),
                    ],
                    dim=-1,
                )

                if bs.pos == len(bs):
                    finished[batch_idx] = True

    return rets


def wrap_bits(
    bits: ConstBitStream,
    size_flag_bits=12,
    ef_flag_bits=4,
    enable_edge_flip=False,
    **kwargs,
) -> ConstBitStream:
    size_bs = ConstBitStream(uint=len(bits), length=size_flag_bits)
    if enable_edge_flip:
        bs_list = [bits]
        for _ in range(1, 2**ef_flag_bits):
            tmp_bs = _ef_encode(bs_list[-1])
            bs_list.append(tmp_bs)
        bs_list_len = [bin(idx).count("1") + count_bs_ones(bs) for idx, bs in enumerate(bs_list)]
        # find the best bs
        best_bs_idx = bs_list_len.index(min(bs_list_len))
        bs = bs_list[best_bs_idx]
        ef_flag = Bits(uint=best_bs_idx, length=ef_flag_bits)
    else:
        ef_flag = Bits(uint=0, length=ef_flag_bits)
        bs = bits

    return size_bs + ef_flag + bs


def unwrap_bits(
    bits: ConstBitStream,
    size_flag_bits=12,
    ef_flag_bits=4,
    **kwargs,
) -> ConstBitStream:
    prefix_len = size_flag_bits + ef_flag_bits
    assert len(bits) >= prefix_len, f"bits length is too short. {len(bits)} < {prefix_len}"
    valid_len = bits.peek(f"uint:{size_flag_bits}")
    ef_flag: int = bits[size_flag_bits:].peek(f"uint:{ef_flag_bits}")

    bs = bits[prefix_len : prefix_len + valid_len]
    for _ in range(ef_flag):
        bs = _ef_decode(bs)

    return ConstBitStream(bs)


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


def count_bs_ones(bs: ConstBitStream) -> int:
    return bs.bin.count("1")


def _huffman_table(freq):
    if freq is not None:
        i2f = {i: f for i, f in enumerate(freq)}
        return huffman.from_frequencies(i2f)
    else:
        return None
