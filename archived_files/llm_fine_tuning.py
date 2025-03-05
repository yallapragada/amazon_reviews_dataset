#!/usr/bin/env python3

"""
llm fine-tuning script with quantization, lora, accelerate
"""

import os
import csv
import glob
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from tqdm.auto import tqdm
import logging
from logging import Logger
import pandas as pd

import torch
from torch.utils.data import DataLoader
import datasets
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    get_scheduler,
    PreTrainedTokenizer,
    PreTrainedModel,
    SchedulerType
)
from bitsandbytes.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.model_selection import train_test_split
from huggingface_hub import login

def load_data(config:dict, logger: Logger):
    
    """Load and preprocess the dataset."""
    logger.info('Loading CSV data using streaming approach')

    data_dir = config['data_dir']
    test_size = float(config['test_size'])
    seed = int(config['seed'])
    data_dir = config['data_dir']
    model_name = config['model_name']
    max_sequence_length = config['max_sequence_length']
    train_batch_size = int(config['train_batch_size'])
    eval_batch_size = int(config['eval_batch_size'])
    
    # Use glob to find all CSV files in the folder
    csv_files = glob.glob(f"{data_dir}/*.csv")
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        raise ValueError(f"No CSV files found in {data_dir}")

    # Read and concatenate all CSV files into a single DataFrame
    df = pd.concat([pd.read_csv(file, engine='python', on_bad_lines='skip', quoting=csv.QUOTE_NONE) for file in csv_files], ignore_index=True)

    df["rating"] = pd.to_numeric(df["rating"], errors='coerce')
    df = df[['rating', 'text']].dropna()
    df["label"] = df["rating"].apply(convert_rating)
    df = df.dropna(subset=["label"])

    cols_to_delete = ['text', '__index_level_0__', 'rating']

    print('Preprocess Data - Split data into train and validation sets')
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)

    print(f'Load tokenizer for {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print('Convert train and validation data to Hugging Face datasets')
    train_dataset = datasets.Dataset.from_pandas(train_df)
    val_dataset = datasets.Dataset.from_pandas(val_df)

    def tokenize_function(examples):
        if isinstance(examples["text"], list):
            examples["text"] = [str(text) for text in examples["text"]]
        else:
            examples["text"] = str(examples["text"])
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_sequence_length)

    print('tokenize data')
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_delete)
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_delete)

    train_dataset.set_format('torch')
    val_dataset.set_format('torch')

    print('data collator with padding a batch of examples to the max length seen in the batch')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, collate_fn=data_collator)
    
    return train_dataloader, eval_dataloader, tokenizer

def format_time(seconds: float) -> str:
    """Return a formatted string HH:MM:SS.ss from seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def convert_rating(x: Union[int, float, str]) -> Optional[int]:
    """Convert numerical rating to binary sentiment label."""
    try:
        return 1 if float(x) >= 3 else 0
    except ValueError:
        # Handle the error case by returning None
        return None


# Setup logger
def setup_logging(log_level: str = "INFO") -> Logger:
    """Set up and return a logger with the specified log level."""
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger
    logger = logging.getLogger("fine_tuning")
    return logger

def main(config: dict) -> None:
    """Main training function."""

    log_level = config.get('log_level', 'INFO')
    compute_dtype = torch.bfloat16
    lora_r = int(config['lora_r'])
    lora_alpha = int(config['lora_alpha'])
    target_modules = config['target_modules'].split(',')
    lora_dropout = float(config['lora_dropout'])
    lora_bias = config['lora_bias']
    learning_rate = float(config['learning_rate'])
    num_epochs = int(config['num_epochs'])
    warmup_ratio = float(config['warmup_ratio'])
    
    # Set up logger
    logger = setup_logging(log_level)

    # Set up device
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    # Login to Hugging Face Hub if token is provided
    if 'hf_token' in config:
        login(token=args.hf_token)
    
    # Load data
    train_dataloader, eval_dataloader, tokenizer = load_data(config, logger)

    # Create the Accelerator instance
    accelerator = Accelerator(mixed_precision='fp8')

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=compute_dtype
    )

    # Load model
    logger.info(f'Loading model: {model_name}')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype
    )
    model.config.pad_token_id = model.config.eos_token_id

    # Apply LoRA Configuration
    logger.info('Applying LoRA Configuration to the Model')
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
    )

    logger.info('Wrapping model with LoRA')
    model = get_peft_model(model, lora_config)

    # Set up optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), learning_rate)
    
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    
    logger.info(f'Length of train_loader = {num_update_steps_per_epoch}')
    logger.info(f'Total training steps = {max_train_steps}')

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(warmup_ratio * max_train_steps),
        num_training_steps=max_train_steps
    )

    # Prepare all objects with Accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    logger.info(f'GPU memory allocated (MB): {torch.cuda.memory_allocated()/(1024*1024):.2f}')

    # Training loop
    logger.info('Starting fine-tuning')
    epoch_progress_bar = tqdm(range(num_epochs), desc="Epochs", colour='red')
    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        batch_progress_bar = tqdm(range(num_update_steps_per_epoch), colour='green')
        
        for batch in train_dataloader:
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass using accelerator
            accelerator.backward(loss)

            # Optimizer and scheduler step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            batch_progress_bar.update(1)
            global_step += 1
        
        epoch_progress_bar.update(1)
    
    logger.info('Fine-tuning complete.')
    
    # Final evaluation
    avg_eval_loss = evaluate(model, eval_dataloader, accelerator, num_epochs, logger)


def evaluate(
    model: PeftModel, 
    eval_dataloader: DataLoader, 
    accelerator: Accelerator, 
    logger: Logger
) -> float:
    """Evaluate the model on the validation set."""
    logger.info(f'Evaluate fine tuned model')
    model.eval()
    total_eval_loss = 0.0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            outputs = model(**batch)
        total_eval_loss += outputs.loss.item()
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    logger.info(f"Validation Loss: {avg_eval_loss:.4f}")
    return avg_eval_loss

def parse_config_file(config_filename):
    """reads a config file and returns a dictionary of arguments"""
    config = {}
    with open(config_filename, "r") as config_file:
        for line in config_file:
            line = line.strip()
            if line and not line.startswith('#'):   #ignore empty lines and comments
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config

if __name__ == "__main__":
    config = parse_config_file(config_filename='llm_fine_tuning.txt')
    main(config)
