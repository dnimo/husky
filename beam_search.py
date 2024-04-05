from transformers import BeamScorer
import torch

def beam_search(
    input_ids: torch.Tensor,
    beam_scorer: BeamScorer,
    logits_processor: None,
    stopping_criteria: None,
    max_length: int,
    pad_token_id: int,
    eos_token_id: int,
    output_attentions: bool,
    output_hidden_states: bool,
    output_scores: bool,
    return_dict_in_generate: bool,
    **model_kwargs
):
    # 初始化
    batch_size, cur_len = input_ids.shape
    