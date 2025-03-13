import os
import torch
import chromadb
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import concurrent.futures
from functools import partial

# Configure ChromaDB Storage Path (local path)
CHROMA_DB_PATH = "./local_chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Create or load a ChromaDB collection
collection = chroma_client.get_or_create_collection(name="document_embeddings")

# Load a HuggingFace dataset (e.g., "ms_marco", "squad", "wiki_qa")
DATASET_NAME = "ms_marco"  # Change to "squad" or "wiki_qa" if needed
MAX_SAMPLES = 100000  # Limit the number of samples for local testing

# Function to load dataset
def load_data():
    try:
        dataset = load_dataset(DATASET_NAME, "v2.1", split="train")
        # Take a subset for local testing
        return dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    except Exception as e:
        print(f"Error loading dataset {DATASET_NAME}: {e}")
        print("Trying to load a smaller subset or different configuration...")
        try:
            # For ms_marco, try the v2.1 version with a specific configuration
            if DATASET_NAME == "ms_marco":
                dataset = load_dataset("ms_marco", "v2.1", split="train")
            else:
                dataset = load_dataset(DATASET_NAME, split="train[:1000]")
            return dataset.select(range(min(MAX_SAMPLES, len(dataset))))
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            raise

# Load a sentence transformer model for embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Function to compute embeddings
def compute_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)  # Mean Pooling
    return outputs.cpu().numpy().tolist()[0]  # Convert to Python list


def query_database(query_text, n_results=5):
    """
    Query the database with a text string and return the most similar documents.
    """
    query_embedding = compute_embedding(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

def process_single_query(sample, query_idx):
    """
    Process a single query and return the results.
    """
    # Skip if the sample doesn't have a query field
    if 'query' not in sample:
        return None
        
    query = sample['query']
    
    # Skip empty queries
    if not query or len(query.strip()) == 0:
        return None
        
    results = query_database(query)
    
    return {
        'query_idx': query_idx,
        'query': query,
        'results': results
    }

def interactive_query(max_workers=10, max_queries=50, start_idx=0, batch_size=None):
    """
    Process queries from the dataset in parallel using multiple threads.
    """
    print("\n=== RAG Query Processing (Parallel) ===")
    print(f"Database contains {collection.count()} documents")
    print(f"Using {max_workers} worker threads")
    
    # Load the dataset directly
    dataset = load_data()
    print(f"Loaded {len(dataset)} samples from {DATASET_NAME}")
    
    # If batch_size is not provided, use max_queries
    if batch_size is None:
        batch_size = max_queries
    
    # Use dataset.select with range to get the appropriate batch
    end_idx = min(start_idx + batch_size, len(dataset))
    samples_to_process = dataset.select(range(start_idx, end_idx))
    
    # Process queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_query, sample, i + start_idx): i + start_idx 
            for i, sample in enumerate(samples_to_process)
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), 
                          total=len(future_to_idx), 
                          desc="Processing queries"):
            result = future.result()
            if result is None:
                continue
                
            query_idx = result['query_idx']
            query = result['query']
            query_results = result['results']
            
            print(f"\nQuery {query_idx+1}: {query}")
            print(f"Top {len(query_results['documents'][0])} results:")
            
            for j, (doc, distance) in enumerate(zip(query_results['documents'][0], query_results['distances'][0])):
                print(f"\n--- Result {j+1} (Similarity: {1-distance:.4f}) ---")
                # Print a preview of the document (first 200 chars)
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                print(preview)
                
            print("\n" + "-"*50)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="RAG Inference with ChromaDB")
    parser.add_argument("--query", action="store_true", help="Query the database interactively")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME, 
                        help=f"HuggingFace dataset name (default: {DATASET_NAME})")
    parser.add_argument("--samples", type=int, default=MAX_SAMPLES,
                        help=f"Maximum number of samples to process (default: {MAX_SAMPLES})")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of worker threads for parallel processing (default: 4)")
    parser.add_argument("--max_queries", type=int, default=100000,
                        help="Maximum number of queries to process (default: 50)")
    
    args = parser.parse_args()
    
    # Update global variables based on args
    DATASET_NAME = args.dataset
    MAX_SAMPLES = args.samples
    
    interactive_query(max_workers=args.workers, max_queries=args.max_queries)
