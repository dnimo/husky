#! -*- coding: utf-8 -*-
# Naive Bayes-based Context Extension (NBCE)

import json
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TopPLogitsWarper, LogitsProcessorList

# 模型路径
model_path = 'cyberagent/open-calm-small'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'left' 
tokenizer.pad_token = tokenizer.unk_token

# 加载LLAMA模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载示例Context
with open('./../data/contexts.json', 'r', encoding='utf-8') as file:
    contexts = json.load(file)

# 示例问题集（一次性问多个问题，NBCE自行根据Context逐一输出答案）
question = """資料をよく読んで、それぞれに答えてください。
- 中国はフィリピン国家電力網公社の株式をどれくらい保有していますか?
- LinkedInは何人の従業員を解雇する予定ですか?
- ギリアドはファーマセットにいくら支払ったのでしょうか?
- C型肝炎の特効薬ソバルディが発売されたのは何年ですか?
- 中央アジアサミットはどこで開催されますか？ 誰が主催?
- 人民軍を侮辱したとして捜査を受けた俳優は誰ですか?
- 「タンクにふさわしい」水路を主張するプロジェクトはどれですか?
- あなたがメルクの CEO だったら、最優先事項は何ですか?"""

# 拼接context和question
contexts = [''] + contexts  # 添加空Context（无Context预测）
batch = ['User: %s\n\n%s\n\nAssistant:' % (context, question) for context in contexts]
print('Context长度分布：', [len(text) for text in batch])
print('Context总长度：', sum([len(text) for text in batch]))

# Top-P截断
processors = LogitsProcessorList()
processors.append(TopPLogitsWarper(0.95))


@torch.no_grad()
def generate(max_tokens):
    """Naive Bayes-based Context Extension 演示代码
    """
    inputs = tokenizer(batch, padding='max_length', max_length=1000, truncation=True, return_tensors='pt').to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    print('input_ids', input_ids.shape)
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

        # 构建分布，采样
        # tau = 1是标准的随机采样，tau->0则是贪心搜索
        # 简单起见，这里没有实现topk、topp截断
        tau = 0.01
        probas = torch.nn.functional.softmax(logits[None] / tau , dim=-1)
        next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)
        if next_tokens[0] == tokenizer.eos_token_id:
            break

        ret = tokenizer.batch_decode(next_tokens)
        print(ret[0], flush=True, end='')

        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)


if __name__ == '__main__':
    generate(10)