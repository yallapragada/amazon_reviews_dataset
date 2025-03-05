import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model

# Load dataset from CSV
csv_file = "Gift_Cards.csv"  
df = pd.read_csv(csv_file, encoding="utf-8")

# Convert rating to integer (binary classification: 1 = positive, 0 = negative)
df["label"] = df["rating"].apply(lambda x: 1 if x >= 3 else 0)
df = df[["label", "text"]] 

# Convert CSV to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Model configuration
model_id = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

# Lora Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)

# Timeout adjustment for downloaded models
# os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'

def preprocess_function(examples):
    texts = examples["text"]
    if isinstance(texts, list):
        texts = [str(t) for t in texts] 
    else:
        texts = [str(texts)]

    tokenizer = tokenizer(
        texts, 
        truncation=True, 
        padding="max_length", 
        max_length=256
    )

    # print(tokenizer)

    return tokenizer

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Split dataset into train and validation
split = encoded_dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
eval_dataset = split["test"]

# Training arguments configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Model training
trainer.train()

# Model evaluation
results = trainer.evaluate()
print(results)

# Save the fine-tuned model
trainer.save_model("./fine_tuned_lora")
tokenizer.save_pretrained("./fine_tuned_lora")
