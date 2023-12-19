import sys

sys.path.append('/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/tools/')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,3,4'

from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM, LineByLineTextDataset,\
                            DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tokenizers import SentencePieceBPETokenizer
from datasets import load_dataset

TokenizerPath = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/kuhpTokenizer/40000_vocab/bert_bpe_vocab_40000.model'
CorpusPath = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/data/del_none_data_for_train_tokenizer.txt'

config = RobertaConfig(
    vocab_size=40000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
)

dataset = load_dataset("text", data_files=CorpusPath)

model = RobertaForMaskedLM(config=config)

tokenizer = SentencePieceBPETokenizer(TokenizerPath)



data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="base",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_gpu_train_batch_size=32,
    save_steps=10000,
    save_total_limit=5,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['text'],
)

trainer.train()

trainer.save_model("kuhp_roberta")