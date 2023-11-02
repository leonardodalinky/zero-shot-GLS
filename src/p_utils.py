import torch


def construct_mask(mask_lens: torch.LongTensor, seq_len: int) -> torch.BoolTensor:
    """
    Construct mask.

    Args:
        mask_lens: (batch_size,) mask len of each batch
        seq_len: seq len of each batch

    Returns:
        torch.BoolTensor: (batch, seq_len) where `True` means left-aligned mask
    """
    assert not mask_lens.is_floating_point()
    assert mask_lens.dim() == 1, "mask_lens must be (batch_size,)."
    tmp = torch.arange(seq_len, device=mask_lens.device).unsqueeze_(0).expand(mask_lens.size(0), -1)
    return tmp < mask_lens.unsqueeze(1)
