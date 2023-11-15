from transformers import BertTokenizer, BertModel
from entropy import calculate_entropy
import lorem
import torch

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 生成长度为10的句子列表
# texts = [lorem.sentence() for _ in range(10)]
texts = []

texts.extend(["world war III", "Obama is dead", "China became a state of the United States","インフルエンザはいつ流行するのですか？","季節性インフルエンザと新型インフルエンザはどう違うのですか？","平成25（2013）年春に中国で発生した、鳥インフルエンザA（H7N9）の現況を教えてください。","平成21（2009）年に流行した、新型インフルエンザの状況を教えてください。"])
print(texts)

# 分词并将文本转换为模型输入格式
# input_ids = tokenizer.encode(text, return_tensors='pt')

# 批量编码
encoded_texts = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt")

# 获取BERT模型的输出
# 获取输入 ID 和 attention mask
input_ids = encoded_texts["input_ids"]
attention_mask = encoded_texts["attention_mask"]

# 将输入批量送入 BERT 模型
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

pool_output = outputs.pooler_output

# 输出形状
print("Pooler output shape:", pool_output.shape)


# compute entropy

en = calculate_entropy(pool_output, batch=True)

print(en)
