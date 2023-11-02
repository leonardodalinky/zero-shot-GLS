"""
Fundamental search algorithms for LLMs.
"""
import torch
import torch.nn.functional as F
import transformers as tr


@torch.no_grad()
def compute_nsp_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Predict the probability of next token.

    Args:
        model: Probably a LLaMa model.
        input_ids: (batch_size, seq_len) tensor of token ids,
            DO NOT add special tokens to the end.
        attention_mask: (batch_size, seq_len) tensor of attention mask.
            Only useful when the batch_size is greater than 2.

    Returns:
        (batch_size, vocab_size) tensor of probabilities.
    """
    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.dim() == 2
    assert attention_mask is None or attention_mask.size() == input_ids.size()
    model: tr.LlamaForCausalLM
    logits: torch.Tensor = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits  # (batch_size, seq_len, vocab_size)
    probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
    if attention_mask is None:
        return probs[:, -1, :]  # (batch_size, vocab_size)
    else:
        valid_len = attention_mask.sum(dim=-1, dtype=torch.long)  # (batch_size,)
        assert (valid_len > 0).all()
        return torch.index_select(probs, dim=1, index=valid_len - 1)  # (batch_size, vocab_size)


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
    model: tr.LlamaForCausalLM

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
    model: tr.LlamaForCausalLM

    generation_config = tr.GenerationConfig.from_model_config(model.config)
    generation_config.length_penalty = -1.0
    generation_config.num_beams = 1
    generation_config.num_return_sequences = 1
    generation_config.eos_token_id = eos_ids
    generation_config.early_stopping = True
    generation_config.max_new_tokens = 32

    ret_ids: torch.Tensor = model.generate(
        input_ids, attention_mask=attention_mask, generation_config=generation_config
    )  # (1, seq_len)

    return ret_ids


@torch.no_grad()
def enhanced_greedy_search(
    model,
    input_ids: torch.Tensor,
    fallback_eos_id: int = 1126,  # token `And` in LLaMa
    capacity: float = 0.6,
    threshold: float = 5e-3,
    max_bits_len: int = 5,
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: Probably a LLaMa model.
        input_ids: (batch_size, seq_len) tensor of token ids,
            DO NOT add special tokens to the end.
        fallback_eos_id: The token id to prevent the model from being stopped.
        capacity: The maximum capacity of the search.
        threshold: The minimum probability of the search.
        max_bits_len: The maximum length of the bits.

    Returns:
        Tuple of:
            (2 ** used_bits, seq_len) tensor of token ids.
            The length of the used bits.
    """
    assert input_ids.dim() == 2
    assert input_ids.size(0) == 1, "Only support batch size 1."
    model: tr.LlamaForCausalLM

    logits: torch.Tensor = model(input_ids=input_ids).logits[0, -1]  # (vocab_size)
    probs = F.softmax(logits, dim=-1)  # (vocab_size)

    sorted_probs, sorted_probs_indice = probs.sort(dim=-1, descending=True, stable=True)
    cum_sorted_probs = torch.cumsum(sorted_probs, dim=-1)

    less_cap_cnt: torch.Tensor = (cum_sorted_probs <= capacity).sum(dtype=torch.long)
    less_thres_cnt: torch.Tensor = (sorted_probs >= threshold).sum(dtype=torch.long)

    trunc_cnt = torch.minimum(less_cap_cnt, less_thres_cnt)
    trunc_cnt = torch.maximum(trunc_cnt, torch.ones_like(trunc_cnt))

    trunc_bits = torch.floor(torch.log2(trunc_cnt.float())).long()
    trunc_bits = torch.minimum(trunc_bits, torch.ones_like(trunc_bits) * max_bits_len)
    trunc_cnt = 2**trunc_bits

    ret_ids = torch.cat(
        [
            input_ids.repeat(trunc_cnt, -1),
            sorted_probs_indice[:trunc_cnt, None],
        ],
        dim=-1,
    )
    # replace every occurence of `eos_id` with `fallback_eos_id`
    generation_config = tr.GenerationConfig.from_model_config(model.config)
    ret_ids[ret_ids == generation_config.eos_token_id] = fallback_eos_id
    ret_ids[ret_ids == generation_config.pad_token_id] = fallback_eos_id
    ret_ids[ret_ids == 2] = fallback_eos_id

    return {
        "comb_ids": ret_ids,
        "trunc_bits": trunc_bits,
        "log_probs": torch.log(sorted_probs[:trunc_cnt].float()),
    }


@torch.no_grad()
def enhanced_greedy_search_end(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    eos_ids: int = [1, 29889, 29973, 29991],
) -> torch.Tensor:
    return beam_search_end(model, input_ids, attention_mask=attention_mask, eos_ids=eos_ids)
