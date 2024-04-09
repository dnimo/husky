from tools import TokenizerPath
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(TokenizerPath)

print(tokenizer.tokenize("今天天氣真好"))


vocab = tokenizer.get_vocab()

print(len(vocab))