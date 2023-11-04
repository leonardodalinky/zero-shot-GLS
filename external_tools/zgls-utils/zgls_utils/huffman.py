from typing import Dict, TypeVar, Union

from ._huffman import huffman_table_from_indices_frequencies

T = TypeVar("T")


def from_frequencies(freqs: Dict[T, Union[int, float]], larger_as_zero=True) -> Dict[T, str]:
    idx2keys = list(freqs.keys())
    indices_freqs = {idx: freqs[k] for idx, k in enumerate(idx2keys)}
    hf_ret = huffman_table_from_indices_frequencies(indices_freqs, larger_as_zero)
    return {idx2keys[idx]: v for idx, v in hf_ret.items()}
