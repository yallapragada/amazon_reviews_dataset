import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from huggingface_hub import login

import pandas as pd

token = ''
login(token=token)

bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

print('Load CSV data into a pandas DataFrame')
df = pd.read_csv('Gift_Cards.csv')
df = df[['rating', 'text']].dropna()
df["label"] = df["rating"].apply(lambda x: 1 if x >= 3 else 0)
#df = df.rename(columns={'rating':'label'})

cols_to_delete = ['text', '__index_level_0__', 'rating']

print('Preprocess Data - Split data into train and validation sets')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print('Load tokenizer for Mistral 7B')
model_name = "mistralai/Mistral-7B-v0.3"  # Use the latest checkpoint of Mistral model on Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
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

print('Load Model for Fine-tuning')
if torch.cuda.is_available():
    device_map = 'cuda:0' 
else:
    device_map = 'cpu'
    
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, device_map='auto', quantization_config=bnb_config)  # Assuming 5 classes for rating
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

print('Define Trainer and TrainingArguments')
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_ratio=0.1,
    weight_decay=0.001,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,  # Evaluate every 100 steps
    save_steps=500,
    save_total_limit=2,
    fp16=True, # Use mixed precision
    gradient_checkpointing=False,
    report_to="none"
)

print('Define Trainer')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer)

print('Start Fine-tuning')
trainer.train()

# Step 7: Save the Fine-tuned Model
model.save_pretrained("./model/fine_tuned_mistral_7b")
tokenizer.save_pretrained("./tokenizer/fine_tuned_mistral_7b")
