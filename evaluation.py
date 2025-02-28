#!/usr/bin/env python3
"""
Text Summarization Evaluation Script

This script evaluates the quality of generated summaries using ROUGE metrics.
It processes CSV files containing both ground truth and predicted summaries.
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import glob
import json
import evaluate
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate text summarization using ROUGE metrics"
    )
    
    # Input/output parameters
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/inference/output_summaries", 
        help="Directory containing summarized CSV files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/inference/evaluation_results", 
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--summary_column", 
        type=str, 
        default="summary", 
        help="Column name for ground truth summaries"
    )
    parser.add_argument(
        "--predicted_column", 
        type=str, 
        default="predicted_summary", 
        help="Column name for predicted summaries"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--use_aggregator", 
        action="store_true", 
        default=True,
        help="Use aggregator for ROUGE calculation"
    )
    
    return parser.parse_args()


def compute_rouge_scores(
    references: List[str], 
    predictions: List[str],
    use_aggregator: bool = True
) -> Dict[str, float]:
    """Compute ROUGE scores between reference and predicted summaries.
    
    Args:
        references: List of reference (ground truth) summaries
        predictions: List of predicted summaries
        use_aggregator: Whether to aggregate scores
        
    Returns:
        Dictionary of ROUGE scores
    """
    # Load the ROUGE metric from the evaluate library
    rouge = evaluate.load('rouge')
    
    # Clean up inputs - handle None or empty strings
    valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
                  if ref and pred and not pd.isna(ref) and not pd.isna(pred)]
    
    if not valid_pairs:
        logger.warning("No valid reference-prediction pairs found")
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "rougeLsum": 0.0
        }
    
    # Unzip the valid pairs
    valid_refs, valid_preds = zip(*valid_pairs)
    
    # Compute ROUGE scores
    results = rouge.compute(
        predictions=valid_preds,
        references=valid_refs,
        use_aggregator=use_aggregator
    )
    
    return results


def evaluate_file(
    file_path: str, 
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Evaluate a single CSV file containing summaries.
    
    Args:
        file_path: Path to the CSV file
        args: Command line arguments
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"Evaluating file: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if required columns exist
    if args.summary_column not in df.columns:
        logger.error(f"Ground truth column '{args.summary_column}' not found in {file_path}")
        return {}
    
    if args.predicted_column not in df.columns:
        logger.error(f"Predicted column '{args.predicted_column}' not found in {file_path}")
        return {}
    
    # Extract reference and predicted summaries
    references = df[args.summary_column].tolist()
    predictions = df[args.predicted_column].tolist()
    
    # Compute ROUGE scores
    scores = compute_rouge_scores(
        references, 
        predictions, 
        use_aggregator=args.use_aggregator
    )
    
    # Add file information to results
    results = {
        "file_name": os.path.basename(file_path),
        "num_samples": len(df),
        "rouge_scores": scores
    }
    
    return results


def main() -> None:
    """Main function to run the evaluation pipeline."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all CSV files in the input directory
    input_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    
    if not input_files:
        logger.warning(f"No CSV files found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(input_files)} CSV files to evaluate")
    
    # Evaluate each file
    all_results = []
    for file_path in input_files:
        results = evaluate_file(file_path, args)
        if results:
            all_results.append(results)
    
    # Calculate average scores across all files
    if all_results:
        # Get all rouge metrics from the first result
        rouge_metrics = list(all_results[0]["rouge_scores"].keys())
        
        # Initialize average scores
        avg_scores = {metric: 0.0 for metric in rouge_metrics}
        
        # Calculate weighted average based on number of samples
        total_samples = 0
        for result in all_results:
            num_samples = result["num_samples"]
            total_samples += num_samples
            
            for metric in rouge_metrics:
                avg_scores[metric] += (
                    result["rouge_scores"][metric] * num_samples
                )
        
        # Normalize by total number of samples
        for metric in rouge_metrics:
            avg_scores[metric] /= total_samples
        
        # Add average scores to results
        summary_results = {
            "total_files": len(all_results),
            "total_samples": total_samples,
            "average_rouge_scores": avg_scores
        }
        
        # Save detailed results
        output_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(output_path, "w") as f:
            json.dump({"file_results": all_results, "summary": summary_results}, f, indent=2)
        
        # Save summary results
        summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_results, f, indent=2)
        
    else:
        logger.warning("No valid results to report")


if __name__ == "__main__":
    main()
