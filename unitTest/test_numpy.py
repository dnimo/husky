import torch

# 假设 logits 是一维数组
logits = torch.tensor([1., 2., 3., 4., 3.8999, 3.9, 3.999])

# 使用 [None] 在数组上添加新轴，并进行逐元素除法
result = logits[None] / 0.4
result = torch.nn.functional.softmax(result, dim=-1)

token = torch.multinomial(result, 1).squeeze(1)
n = logits.shape[0]
input_ids = token.unsqueeze(-1).tile(n, 1)
# 打印结果
print(result)

print(token)

print(input_ids)