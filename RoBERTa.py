from transformers import AutoTokenizer, RobertaConfig, RobertaForMaskedLM, LineByLineTextDataset,\
                            DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tools import TokenizerPath
from transformers.tokenization_utils import PreTrainedTokenizer

config = RobertaConfig(
    vocab_size=250315,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
)

tokenizer = AutoTokenizer.from_pretrained(TokenizerPath, max_len=config.max_position_embeddings)

model = RobertaForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=r"C:\Users\KuoChing\workspace\husky\data\sample.txt",
    block_size=96,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="kuhp_roberta/checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=10000,
    save_total_limit=5,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("kuhp_roberta/model")