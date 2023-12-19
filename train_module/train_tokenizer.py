from tokenizers import Tokenizer, models, trainers, processors
from datasets import load_dataset

# 载入你的文本数据
# text_data = ["Your text data goes here.", "Another example sentence."]

CorpusPath = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/data/del_none_data_for_train_tokenizer.txt'

text_data = load_dataset("text", data_files=CorpusPath, split='train')


text_data = text_data['text']

# 初始化 Tokenizer
tokenizer = Tokenizer(models.BPE())

# 自定义训练参数
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"], vocab_size=50000)

#iterator
def batch_iterator(batch_size=1000):
    for i in range(0, len(text_data), batch_size):
        yield text_data[i : i + batch_size]

# 拟合 Tokenizer
tokenizer.train_from_iterator(batch_iterator(),trainer=trainer, length=len(text_data))

# 保存 Tokenizer 模型
tokenizer.save("kuhp_bert_tokenizer.json")