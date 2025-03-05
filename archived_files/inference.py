#!/usr/bin/env python3
"""
Text Summarization Inference Script

This script performs text summarization using a pre-trained language model.
It can process multiple CSV files, summarize the text in them, and save the results.
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
        self.dataset = load_dataset("csv", data_files=csv_file, streaming=True)

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run text summarization inference with a language model"
    )
    
    # Dataset parameters
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/inference/data", 
        help="Directory for input CSV files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/inference/output_summaries", 
        help="Directory for output summary files"
    )
    parser.add_argument(
        "--file_size_kb", 
        type=int, 
        default=100, 
        help="Target size in KB for each dataset chunk"
    )
    parser.add_argument(
        "--max_files", 
        type=int, 
        default=20, 
        help="Maximum number of dataset files to create"
    )
    
    # Model parameters
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="mistralai/Mistral-7B-v0.1", 
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None, 
        help="Hugging Face API token"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for inference"
    )
    parser.add_argument(
        "--text_column", 
        type=str, 
        default="text", 
        help="Column name in CSV containing text to summarize"
    )
    
    # Inference parameters
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512, 
        help="Maximum input sequence length"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100, 
        help="Maximum number of tokens to generate for summary"
    )
    
    # Quantization parameters
    parser.add_argument(
        "--load_in_4bit", 
        action="store_true", 
        help="Load model in 4-bit precision"
    )
    
    parser.add_argument(
        "--quant_type", 
        type=str, 
        default="nf4", 
        choices=["nf4", "fp4"], 
        help="Quantization type for 4-bit quantization"
    )
    
    parser.add_argument(
        "--use_double_quant", 
        action="store_true", 
        help="Use double quantization"
    )
    
    parser.add_argument(
        "--compute_dtype", 
        type=str, 
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Compute dtype for 4-bit quantization"
    )
    
    
    return parser.parse_args()


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


def setup_model_and_tokenizer(args: argparse.Namespace) -> tuple:
    """Set up the model and tokenizer.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (tokenizer, model, accelerator)
    """
    # Initialize Hugging Face Accelerator
    accelerator = Accelerator()

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_compute_dtype=getattr(torch, args.compute_dtype),
        bnb_4bit_use_double_quant=args.use_double_quant,
    )

    # Load model with automatic device placement (uses multiple GPUs if available)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()  # Set model to evaluation mode

    # Wrap model for distributed inference with Accelerate
    model = accelerator.prepare(model)
    
    return tokenizer, model, accelerator


def process_csv_file(
    csv_file: str, 
    args: argparse.Namespace, 
    tokenizer: Any, 
    model: Any, 
    accelerator: Accelerator
) -> None:
    """Process a single CSV file and generate summaries.
    
    Args:
        csv_file: Name of the CSV file to process
        args: Command line arguments
        tokenizer: Tokenizer for the model
        model: Language model for summarization
        accelerator: Accelerator for distributed inference
    """
    input_path = os.path.join(args.data_dir, csv_file)
    output_path = os.path.join(args.output_dir, f"summarized_{csv_file}")

    logger.info(f"Processing {csv_file}...")

    # Read the original CSV file
    original_df = pd.read_csv(input_path)
    
    dataset = CSVDataset(
        input_path, 
        text_column=args.text_column, 
        batch_size=args.batch_size
    )

    summaries = []
    for batch in dataset:
        summarized_texts = summarize_text(
            batch, 
            tokenizer, 
            model, 
            accelerator,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens
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
    args = parse_args()
    
    # Login to Hugging Face if token is provided
    if args.hf_token:
        login(token=args.hf_token)
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up model and tokenizer
    tokenizer, model, accelerator = setup_model_and_tokenizer(args)

    # Get dataset files
    csv_files = load_dataset_files(args.data_dir, args.file_size_kb, args.max_files)

    # Process each CSV file
    csv_files = [f for f in os.listdir(args.data_dir) if f.endswith(".csv")]

    for csv_file in csv_files:
        process_csv_file(csv_file, args, tokenizer, model, accelerator)

    logger.info("Summarization complete for all files.")


if __name__ == "__main__":
    main()
