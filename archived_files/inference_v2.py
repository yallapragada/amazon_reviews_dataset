#!/usr/bin/env python3
"""
Text Summarization Inference Script

This script performs text summarization using a pre-trained language model.
It can process multiple CSV files, summarize the text in them, and save the results.
Configuration is loaded from a config file.
"""

import os
import logging
import argparse
from typing import List, Dict, Any
import uuid

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from torch.utils.data import Dataset
from datasets import load_dataset
from huggingface_hub import login


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CSVDataset(Dataset):
    """Dataset class for streaming CSV reading.
    
    Provides batched iteration over CSV files for efficient processing.
    """
    
    def __init__(self, csv_file: str, text_column: str = "document", batch_size: int = 10):
        """Initialize the dataset.
        
        Args:
            csv_file: Path to the CSV file
            text_column: Name of the column containing text to summarize
            batch_size: Number of samples to process at once
        """
        self.csv_file = csv_file
        self.text_column = text_column
        self.batch_size = batch_size
        self.dataset = load_dataset("csv", data_files=csv_file, streaming=True, split="train")
        

    def __iter__(self):
        """Iterate over the dataset in batches."""
        batch = []
        for sample in self.dataset:
            batch.append(sample[self.text_column])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:  # Yield remaining batch
            yield batch


def parse_config_file(config_filename):
    """Reads a config file and returns a dictionary of configuration parameters.
    
    Args:
        config_filename: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config = {}
    with open(config_filename, "r") as config_file:
        for line in config_file:
            line = line.strip()
            if line and not line.startswith('#'):  # Ignore empty lines and comments
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    
    # Convert values to appropriate types
    type_conversions = {
        'data_dir': str,
        'output_dir': str,
        'file_size_kb': int,
        'max_files': int,
        'model_name': str,
        'hf_token': lambda x: None if x.lower() == 'none' else str(x),
        'batch_size': int,
        'text_column': str,
        'max_length': int,
        'max_new_tokens': int,
        'load_in_4bit': lambda x: x.lower() == 'true',
        'quant_type': str,
        'use_double_quant': lambda x: x.lower() == 'true',
        'compute_dtype': str
    }
    
    for key, type_func in type_conversions.items():
        if key in config:
            config[key] = type_func(config[key])
    
    return config


def load_dataset_files(data_dir: str, file_size_kb: int = 100, max_files: int = 20) -> List[str]:
    """Load or create dataset files.
    
    Args:
        data_dir: Directory to store dataset files
        file_size_kb: Target size in KB for each dataset chunk
        max_files: Maximum number of files to create
        
    Returns:
        List of paths to dataset files
    """
    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if the dataset is already downloaded
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
    
    if csv_files:
        logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
        return csv_files
    else:
        logger.info(f"No CSV files found in {data_dir}")
        logger.info(f"Building dataset with files of approximately {file_size_kb}KB each")
        return build_dataset(data_dir, file_size_kb, max_files)


def build_dataset(data_dir: str, file_size_kb: int = 100, max_files: int = 20) -> List[str]:
    """Build dataset by downloading and splitting into multiple files.
    
    Args:
        data_dir: Directory to store dataset files
        file_size_kb: Target size in KB for each dataset chunk
        max_files: Maximum number of files to create
        
    Returns:
        List of paths to created dataset files
    """
    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a directory for the full dataset
    full_dataset_dir = os.path.join(data_dir, "full_dataset")
    os.makedirs(full_dataset_dir, exist_ok=True)
    
    # Download the dataset
    dataset = load_dataset("kqsong/OASum", split="test")
    
    # Save the full dataset to the full_dataset directory
    full_path = os.path.join(full_dataset_dir, "train_full.csv")
    dataset.to_csv(full_path)
    logger.info(f"Saved full dataset to {full_path}")
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset)
    
    # Add a UUID column to each row
    df['uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Create a list to store the paths of chunked files
    csv_paths = []
    
    # Calculate approximate rows per file based on file size
    # First, save a small sample to estimate size per row
    sample_size = min(1000, len(df))
    sample_df = df.head(sample_size)
    sample_path = os.path.join(data_dir, "sample.csv")
    sample_df.to_csv(sample_path, index=False)
    
    # Get file size in KB
    sample_size_kb = os.path.getsize(sample_path) / 1024
    # Calculate rows per KB
    rows_per_kb = sample_size / sample_size_kb
    # Calculate rows per target file size
    rows_per_file = int(rows_per_kb * file_size_kb)
    
    # Remove the sample file
    os.remove(sample_path)
    
    # Adjust rows_per_file to ensure we don't exceed max_files
    total_rows = len(df)
    if rows_per_file > 0:
        num_files_needed = total_rows / rows_per_file
        if num_files_needed > max_files:
            rows_per_file = int(total_rows / max_files) + 1
    else:
        rows_per_file = int(total_rows / max_files) + 1
    
    logger.info(
        f"Creating approximately {min(max_files, total_rows // rows_per_file + 1)} "
        f"files with ~{rows_per_file} rows each"
    )
    
    # Split the dataset into files of the calculated size
    for i in range(0, total_rows, rows_per_file):
        if len(csv_paths) >= max_files:
            break
            
        end_idx = min(i + rows_per_file, total_rows)
        chunk = df.iloc[i:end_idx]
        chunk_path = os.path.join(data_dir, f"dataset_chunk_{i}_{end_idx}.csv")
        chunk.to_csv(chunk_path, index=False)
        csv_paths.append(chunk_path)
        
        # Check file size
        actual_size_kb = os.path.getsize(chunk_path) / 1024
        logger.info(f"Created file {len(csv_paths)}: {chunk_path} ({actual_size_kb:.2f}KB)")

    logger.info(f"Created {len(csv_paths)} CSV files in {data_dir}")
    return csv_paths


def summarize_text(
    texts: List[str], 
    tokenizer: Any, 
    model: Any, 
    accelerator: Accelerator, 
    max_length: int = 512, 
    max_new_tokens: int = 100
) -> List[str]:
    """Run summarization inference on a batch of texts.
    
    Args:
        texts: List of texts to summarize
        tokenizer: Tokenizer for the model
        model: Language model for summarization
        accelerator: Accelerator for distributed inference
        max_length: Maximum input sequence length
        max_new_tokens: Maximum number of tokens to generate for summary
        
    Returns:
        List of summarized texts
    """
    # Add a summarization prompt to each text
    prompts = [f"Summarize the following text:\n{text}\nSummary:" for text in texts]

    # Tokenize input texts
    inputs = tokenizer(
        prompts, 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    inputs = {key: val.to(accelerator.device) for key, val in inputs.items()}

    # Generate summaries with the model
    with torch.no_grad():
        summaries = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return [tokenizer.decode(s, skip_special_tokens=True) for s in summaries]


def setup_model_and_tokenizer(config: Dict[str, Any]) -> tuple:
    """Set up the model and tokenizer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (tokenizer, model, accelerator)
    """
    # Initialize Hugging Face Accelerator
    accelerator = Accelerator()

    # Load model and tokenizer
    logger.info(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['load_in_4bit'],
        bnb_4bit_quant_type=config['quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, config['compute_dtype']),
        bnb_4bit_use_double_quant=config['use_double_quant'],
    )

    # Load model with automatic device placement (uses multiple GPUs if available)
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()  # Set model to evaluation mode

    # Move model to device but don't wrap with accelerator.prepare()
    model = model.to(accelerator.device)
    
    return tokenizer, model, accelerator


def process_csv_file(
    csv_file: str, 
    config: Dict[str, Any], 
    tokenizer: Any, 
    model: Any, 
    accelerator: Accelerator
) -> None:
    """Process a single CSV file and generate summaries.
    
    Args:
        csv_file: Name of the CSV file to process
        config: Configuration dictionary
        tokenizer: Tokenizer for the model
        model: Language model for summarization
        accelerator: Accelerator for distributed inference
    """
    input_path = os.path.join(config['data_dir'], csv_file)
    output_path = os.path.join(config['output_dir'], f"summarized_{csv_file}")

    logger.info(f"Processing {csv_file}...")

    # Read the original CSV file
    original_df = pd.read_csv(input_path)
    
    dataset = CSVDataset(
        input_path, 
        text_column=config['text_column'], 
        batch_size=config['batch_size']
    )

    summaries = []
    for batch in dataset:
        summarized_texts = summarize_text(
            batch, 
            tokenizer, 
            model, 
            accelerator,
            max_length=config['max_length'],
            max_new_tokens=config['max_new_tokens']
        )
        summaries.extend(summarized_texts)

    # Make sure we have the right number of summaries
    if len(summaries) != len(original_df):
        logger.warning(
            f"Number of summaries ({len(summaries)}) doesn't match number of rows in original file ({len(original_df)}). "
            f"Truncating to shorter length."
        )
        # Truncate to the shorter length
        min_length = min(len(summaries), len(original_df))
        summaries = summaries[:min_length]
        original_df = original_df.iloc[:min_length]

    # Add the predicted summaries to the original dataframe
    original_df["predicted_summary"] = summaries
    
    # Save the enhanced dataframe to a new CSV file
    original_df.to_csv(output_path, index=False)

    logger.info(f"Saved original data with summaries to {output_path}")


def main() -> None:
    """Main function to run the summarization pipeline."""
    # Parse command line argument for config file path
    parser = argparse.ArgumentParser(description="Run text summarization with config file")
    parser.add_argument("--config", type=str, default="inference_config.txt", help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration from file
    config = parse_config_file(args.config)
    
    # Login to Hugging Face if token is provided
    if config['hf_token']:
        login(token=config['hf_token'])
    
    # Create directories
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)

    # Set up model and tokenizer
    tokenizer, model, accelerator = setup_model_and_tokenizer(config)

    # Get dataset files
    csv_files = load_dataset_files(config['data_dir'], config['file_size_kb'], config['max_files'])

    # Process each CSV file
    csv_files = [f for f in os.listdir(config['data_dir']) if f.endswith(".csv")]

    for csv_file in csv_files:
        process_csv_file(csv_file, config, tokenizer, model, accelerator)

    logger.info("Summarization complete for all files.")


if __name__ == "__main__":
    main()
