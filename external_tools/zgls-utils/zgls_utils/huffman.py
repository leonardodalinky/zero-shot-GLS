from typing import Any, Dict, Union

from ._huffman import huffman_table_from_indices_frequencies


def from_frequencies(freqs: Dict[Any, Union[int, float]], larger_as_zero=True) -> Dict[Any, str]:
    idx2keys = list(freqs.keys())
    indices_freqs = {idx: freqs[k] for idx, k in enumerate(idx2keys)}
    hf_ret = huffman_table_from_indices_frequencies(indices_freqs, larger_as_zero)
    return {idx2keys[idx]: v for idx, v in hf_ret.items()}
