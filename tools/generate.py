import torch
import math
import sys
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM
from transformers import TopPLogitsWarper, LogitsProcessorList, TopKLogitsWarper
from peft import PeftModel
from ParallelContextsWindows.pcw_wrapper import PCWModelWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

Calm_Window_Size = 2048
model_path = "cyberagent/open-calm-small"
peftmodel_path = ''


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def pcw_generate(model, tokenizer, context_texts: list[str], task_texts: str, context_window_size,
                 right_indentation=False):
    model = PCWModelWrapper(model, tokenizer, device, context_window_size, right_indentation)
    outputs = model.pcw_generate(context_texts, task_texts)
    return "".join(outputs)


def nbce_generate(input_ids, attention_mask, model, tokenizer, max_new_tokens=800):
    processors = LogitsProcessorList([TopPLogitsWarper(0.95)])
    n = input_ids.shape[0]
    preds = []
    past_key_values = None
    for i in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values

        beta = 0.45
        eta = 0.1
        logits = outputs.logits[:, -1]
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        logits = processors(input_ids, logits)
        entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
        if i > 0:
            entropy[k] -= eta
        k = entropy[1:].argmin() + 1
        logits_max = -logits[k]
        logits_uncond = logits[0]
        logits_merge = -(1 + beta) * logits_max - beta * logits_uncond
        logits = torch.where(logits_uncond > -100, logits_merge, logits_max)

        probs = logits.softmax(dim=-1)
        new_token = torch.multinomial(probs, 1).sequeeze(1)
        if new_token == tokenizer.eos_token_id:
            break
        ret = tokenizer.bath_decode(new_token)
        print(ret[0], flush=True, end="")

        preds.append(ret[0])
        input_ids = new_token.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, device=model.device)], dim=1)


def longlora_generate():
    pass


if __name__ == "__main__":
    context_texts = ["The quick brown fox jumps over the lazy dog."]
    task_texts = "The quick brown fox jumps over the lazy dog."
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = load_tokenizer(model_path)
    # model = PeftModel.from_pretrained(model, peftmodel_path, torch_dtype=torch.float16, offload_folder=None)
    res = pcw_generate(model, tokenizer, context_texts, task_texts, Calm_Window_Size, right_indentation=False)

    # print(res, flush=True, end="")
