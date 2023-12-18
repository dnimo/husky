from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM, LineByLineTextDataset,\
                            DataCollatorForLanguageModeling, Trainer, TrainingArguments

config = RobertaConfig(
    vocab_size=40000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large', max_len=config.max_position_embeddings)

model = RobertaForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=r"C:\Users\KuoChing\workspace\husky\data\sample.txt",
    block_size=1,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_gpu_train_batch_size=1,
    save_steps=1,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("kuhp_roberta")