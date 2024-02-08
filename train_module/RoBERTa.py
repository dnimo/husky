from transformers import AutoTokenizer, RobertaConfig, RobertaForMaskedLM, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tools import TokenizerPath
from datasets import load_dataset

config = RobertaConfig(
    vocab_size=250315,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
)

tokenizer = AutoTokenizer.from_pretrained(TokenizerPath, max_len=config.max_position_embeddings)

model = RobertaForMaskedLM(config=config)

data = load_dataset("text", data_files=r"C:\Users\KuoChing\workspace\husky\data\sample.txt", split="train")

train_dataset = data.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="kuhp_roberta/checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10000,
    save_total_limit=5,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()

trainer.save_model("kuhp_roberta/model")
