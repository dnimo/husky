#! -*- coding: utf-8 -*-
# Naive Bayes-based Context Extension (NBCE)

import torch
from transformers import TopPLogitsWarper, LogitsProcessorList

# Top-P截断
processors = LogitsProcessorList()
processors.append(TopPLogitsWarper(0.95))


@torch.no_grad()
def generate(max_tokens, tokenizer, model, device, batch):
    """Naive Bayes-based Context Extension 演示代码
    """
    inputs = tokenizer(batch).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    past_key_values = None
    n = input_ids.shape[0]

    for i in range(max_tokens):
        # 模型输出
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        use_cache=True,
                        past_key_values=past_key_values
                        )
        past_key_values = outputs.past_key_values

        # ===== 核心代码开始 =====
        beta, eta = 0.25, 0.1
        logits = outputs.logits[:, -1]
        logits = logits - logits.logsumexp(dim=-1, keepdims=True)
        logits = processors(input_ids, logits)
        entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
        if i > 0:
            entropy[k] -= eta
        k = entropy[1:].argmin() + 1
        logits_max = logits[k]
        logits_uncond = logits[0]
        logits_merged = (1 + beta) * logits_max - beta * logits_uncond
        logits = torch.where(logits_uncond > -100, logits_merged, logits_max)
        # ===== 核心代码结束 =====

        tau = 0.01
        probas = torch.nn.functional.softmax(logits[None] / tau, dim=-1)
        next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)
        if next_tokens[0] == tokenizer.eos_token_id:
            break

        ret = tokenizer.batch_decode(next_tokens)
        print(ret[0], flush=True, end='')

        # prepare for the next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)
