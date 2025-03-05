
import csv
import glob
import torch
from torch.utils.data import DataLoader
import datasets
from accelerate import Accelerator
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BitsAndBytesConfig, 
    DataCollatorWithPadding,
    get_scheduler
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from huggingface_hub import login

import pandas as pd
import time

start_time = time.time()

def format_time(seconds):     
  """Return a formatted string HH:MM:SS.ss from seconds."""    
  hours = int(seconds // 3600)     
  minutes = int((seconds % 3600) // 60)     
  secs = seconds % 60
  return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

token = ''
login(token=token)

# Create the Accelerator instance.
# We can enable mixed precision and gradient accumulation if needed.
accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=1)

bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

print('Load CSV data into a pandas DataFrame')

# Specify the folder path where your CSV files are stored
csv_folder_path = "data"  # <-- Update this with your actual folder path

# Use glob to find all CSV files in the folder
csv_files = glob.glob(f"{csv_folder_path}/*.csv")
if not csv_files:
    raise ValueError("No CSV files found in the specified folder.")

# Read and concatenate all CSV files into a single DataFrame
df = pd.concat([pd.read_csv(file, engine='python', on_bad_lines='skip', quoting=csv.QUOTE_NONE) for file in csv_files], ignore_index=True)

def convert_rating(x):
    try:
        return 1 if x >= 3 else 0
    except ValueError:
        # Handle the error case, for example, by assigning a default label or skipping the row
        return None  # or a default value, such as 0 or 1

df["rating"] = pd.to_numeric(df["rating"], errors='coerce')
df = df[['rating', 'text']].dropna()
df["label"] = df["rating"].apply(convert_rating)
df = df.dropna(subset=["label"])  # If you returned None for problematic rows

cols_to_delete = ['text', '__index_level_0__', 'rating']

print('Preprocess Data - Split data into train and validation sets')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print('Load tokenizer for Mistral 7B')
model_name = "mistralai/Mistral-7B-v0.3"  # Use the latest checkpoint of Mistral model on Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):
    if isinstance(examples["text"], list):
        examples["text"] = [str(text) for text in examples["text"]]
    else:
        examples["text"] = str(examples["text"])
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

print('Convert train and validation data to Hugging Face datasets')
train_dataset = datasets.Dataset.from_pandas(train_df)
val_dataset = datasets.Dataset.from_pandas(val_df)

print('tokenize data')
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_delete)
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_delete)

train_dataset.set_format('torch')
val_dataset.set_format('torch')

print(train_dataset[0])
print(val_dataset[0])

print('data collator with padding a batch of examples to the max length seen in the batch')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)
eval_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=data_collator)
    
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2, 
    device_map='auto', 
    quantization_config=bnb_config
)  
model.config.pad_token_id = model.config.eos_token_id

print('Apply LoRA Configuration to the Model')
lora_config = LoraConfig(
    r=8,  # Low-rank adapter size
    lora_alpha=32,  # Scaling factor for LoRA
    target_modules=["q_proj", "v_proj"],  # Specify which layers to apply LoRA to
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # No bias
)

print('Wrap model with LoRA')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Set up optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
max_train_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * max_train_steps),
    num_training_steps=max_train_steps
)

# Prepare all objects with Accelerator.
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

print('Start Fine-tuning')
progress_bar = tqdm(range(max_train_steps))
global_step = 0
model.train()

for epoch in range(num_train_epochs):
    for batch in train_dataloader:
        # Forward pass. Note: no need to call .to(device) on batch as Accelerator handles it.
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass using accelerator
        accelerator.backward(loss)
        
        # Optimizer and scheduler step
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        global_step += 1
        
print('Fine-tuning complete.')

# Evaluate the model on the validation set
model.eval()
total_eval_loss = 0.0
for batch in eval_dataloader:
    with torch.no_grad():
        outputs = model(**batch)
    total_eval_loss += outputs.loss.item()
avg_eval_loss = total_eval_loss / len(eval_dataloader)
print(f"Validation Loss: {avg_eval_loss}")

# Save the Fine-tuned Model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if accelerator.is_main_process:
    unwrapped_model.save_pretrained("./model/fine_tuned_mistral_7b", save_function=accelerator.save)
    tokenizer.save_pretrained("./tokenizer/fine_tuned_mistral_7b")

end_time = time.time()
print(f"Training completed in: {format_time(end_time-start_time)}")
