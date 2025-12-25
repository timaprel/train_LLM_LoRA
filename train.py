import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
import argparse
import os

parser = argparse.ArgumentParser(description="Обучение TinyLlama")
parser.add_argument("--csv_file", type=str, required=True, help="Путь к CSV файлу")
parser.add_argument("--num_rows", type=int, default=None, help="Количество строк")
parser.add_argument("--output_dir", type=str, default="./tinllama-lora-electronics", help="Директория для сохранения")
args = parser.parse_args()

df = pd.read_csv(args.csv_file, encoding='utf-8')

if args.num_rows is not None:
    df = df.head(min(args.num_rows, len(df)))

prompts = []
for _, row in df.iterrows():
    prompt = f"<|user|>\n{row['question']}</s>\n<|assistant|>\n{row['answer']}</s>\n"
    prompts.append(prompt)

dataset = Dataset.from_dict({"text": prompts})
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    dtype=torch.float16,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=2,
    logging_first_step=True,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=collator,
)

trainer.train()

trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

import matplotlib.pyplot as plt

log_history = trainer.state.log_history
train_steps = []
train_losses = []
eval_steps = []
eval_losses = []

for log in log_history:
    if "loss" in log and "eval_loss" not in log:
        train_steps.append(log.get("step"))
        train_losses.append(log.get("loss"))
    if "eval_loss" in log:
        eval_steps.append(log.get("step"))
        eval_losses.append(log.get("eval_loss"))

plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_losses, label="Train loss")
plt.plot(eval_steps, eval_losses, label="Eval loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("График обучения TinyLlama (LoRA)")
plt.legend()
plt.grid(True)
plt.show()

print("Обучение завершено!")
