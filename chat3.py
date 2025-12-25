import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse

# Установка seed
SEED = 1203
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Парсер аргументов
parser = argparse.ArgumentParser(description="Общение с TinyLlama")
parser.add_argument("--use_lora", action="store_true", help="Использовать LoRA адаптер")
parser.add_argument("--lora_path", type=str, default="./tinllama-lora-electronics", help="Путь к LoRA")
parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Модель")
args = parser.parse_args()

# Устройство
device = "cuda" if torch.cuda.is_available() else "cpu"

# Конфиг для 4-бит
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

# LoRA
if args.use_lora:
    model = PeftModel.from_pretrained(model, args.lora_path)

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token

model.eval()

# Функция общения
def chat(prompt, max_new_tokens=200, temperature=0.7):
    # Восстановление seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Форматирование промпта
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Удаляем промпт из ответа
    if formatted_prompt in response:
        response = response.replace(formatted_prompt, "").strip()
    
    return response

# Интерактивный режим
print("Введите 'exit' для выхода")
while True:
    prompt = input("\nВы: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    response = chat(prompt)
    print("\nМодель:", response)
