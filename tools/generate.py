import torch
import math
import sys
import numpy as np
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


def nbce_generate(model, tokenizer, context_texts: list[str], task_texts: str, max_new_tokens=800):
    processors = LogitsProcessorList([TopPLogitsWarper(0.95)])
    inputs = []
    for line in context_texts:
        inputs.append(line + task_texts)
    inputs.append(task_texts)
    inputs = tokenizer(inputs, return_tensors="pt", padding='max_length', truncation=True, max_length=1024,
                       return_attention_mask=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
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
        eta = 0.2
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

        probs = torch.nn.functional.softmax(logits[None], dim=-1)
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
        print(probs, flush=True, end="")
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


def fisher_generate(theta, epsilon=1e-6):
    """
        Calculate the Fisher Information Matrix.

        Parameters:
        log_likelihood_func: function
            The log likelihood function.
        theta: array-like
            The parameters at which to evaluate the Fisher Information.
        epsilon: float
            A small number for numerical differentiation.

        Returns:
        fisher_info_matrix: 2D array
            The Fisher Information Matrix.
        """
    theta = np.array(theta)
    num_params = len(theta)
    fisher_info_matrix = np.zeros((num_params, num_params))

    for i in range(num_params):
        for j in range(num_params):
            theta_eps_i = np.array(theta, copy=True)
            theta_eps_j = np.array(theta, copy=True)
            theta_eps_i[i] += epsilon
            theta_eps_j[j] += epsilon


    return fisher_info_matrix


if __name__ == "__main__":
    context_texts = ["ZHANGは山に登るのがとても好きです", "ZHANGさんは京都大学に在学中です",
                     "ZHANGさんは中国山東省出身です"]
    task_texts = "張さんの紹介をお願いします"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = load_tokenizer(model_path)
    # model = PeftModel.from_pretrained(model, peftmodel_path, torch_dtype=torch.float16, offload_folder=None)
    # res = pcw_generate(model, tokenizer, context_texts, task_texts, Calm_Window_Size, right_indentation=False)
    #
    # print(res, flush=True, end="")

    res_nbce = nbce_generate(model, tokenizer, context_texts, task_texts, Calm_Window_Size)
    print('\n')
    print(res_nbce, flush=True, end="")
