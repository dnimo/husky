import numpy as np
from transformers import TopPLogitsWarper, LogitsProcessorList
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM
import torch
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

Calm_Window_Size = 2048
model_path = "cyberagent/open-calm-small"
peftmodel_path = ''


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def fisher_information(log_likelihood_func, outputs, input_ids, epsilon=1e-5):
    """
    计算Fisher信息矩阵。

    参数:
    log_likelihood_func: function
        对数似然函数。
    past_key_value: array-like
        大语言模型的past_key_value。
    epsilon: float
        用于数值微分的小量。

    返回:
    fisher_info_matrix: 2D array
        Fisher信息矩阵。
    """
    last_past_key_values = outputs.past_key_values[-1][1]
    num_params = sum(pkv.numel() for pkv in last_past_key_values)
    fisher_info_matrix = np.zeros((num_params, num_params))
    size_of_epsilon = sys.getsizeof(epsilon)

    for i in range(num_params):
        for j in range(num_params):
            # 增加epsilon到第i个参数
            perturbed_pkv_i = [pkv.clone().detach for pkv in last_past_key_values]
            perturbed_pkv_i_flat = perturbed_pkv_i[i // size_of_epsilon]
            perturbed_pkv_i_flat[i % size_of_epsilon] += epsilon

            # 增加epsilon到第j个参数
            perturbed_pkv_j = [pkv.clone().detach for pkv in last_past_key_values]
            perturbed_pkv_j_flat = perturbed_pkv_j[j // size_of_epsilon]
            perturbed_pkv_j_flat[j % size_of_epsilon] += epsilon

            # 计算对数似然函数值
            log_likelihood_i = log_likelihood_func(perturbed_pkv_i, input_ids)
            log_likelihood_j = log_likelihood_func(perturbed_pkv_j, input_ids)
            log_likelihood_ij = log_likelihood_func(last_past_key_values, input_ids)

            # 计算二阶导数的近似值
            d2_log_likelihood = (log_likelihood_i + log_likelihood_j - 2 * log_likelihood_ij) / (epsilon ** 2)

            # 将结果放入Fisher信息矩阵
            fisher_info_matrix[i, j] = d2_log_likelihood

    return fisher_info_matrix


def log_likelihood_GPT_NeoX(input_ids, past_key_value):
    # 这里定义你的对数似然函数
    # 这个函数应该返回给定past_key_value的对数似然值
    processors = LogitsProcessorList([TopPLogitsWarper(0.95)])
    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=past_key_value)
        logits = outputs.logits[:, -1]
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        logits = processors(input_ids, logits)
        probs = torch.nn.functional.softmax(logits[None], dim=-1)

        target_log_probs = probs[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        log_likelihood = target_log_probs.sum()

    return log_likelihood


if __name__ == "__main__":

    context_texts = ["ZHANGは山に登るのがとても好きです", "ZHANGさんは京都大学に在学中です",
                     "ZHANGさんは中国山東省出身です"]
    task_texts = "張さんの紹介をお願いします"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = load_tokenizer(model_path)

    inputs = []

    for line in context_texts:
        inputs.append(line + task_texts)
    inputs.append(task_texts)
    inputs = tokenizer(inputs, return_tensors="pt", padding='max_length', truncation=True, max_length=1024,
                       return_attention_mask=True)

    input_ids = inputs.input_ids.to(device)
    print(input_ids.size())
    attention_mask = inputs.attention_mask.to(device)
    past_key_values = None
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        use_cache=True,
        past_key_values=past_key_values,
    )

    fim = fisher_information(log_likelihood_GPT_NeoX, outputs, input_ids)
    print("Fisher信息矩阵:")
    print(fim)
