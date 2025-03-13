# load and search ChromaDB index

import os
import torch
import chromadb
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from accelerate import Accelerator

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

accelerator = Accelerator()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(accelerator.device)
print(f"Using device: {device}")

# Function to compute embeddings
def compute_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)  # Mean Pooling
    return outputs.cpu().numpy().tolist()[0]  # Convert to Python list

def store_embeddings():
    """
    Generates embeddings and stores them in ChromaDB.
    """
    dataset = load_data()
    print(f"Loaded {len(dataset)} samples from {DATASET_NAME}")
    
    # Check dataset structure
    print("Dataset features:", dataset.features)
    sam = dataset[0]
    print("Sample keys:", sam.keys())
    print(type(dataset))
    print("Example sample:", sam)
    
    # Determine which field to use for text
    text_field = "answers"
    
    print(f"Using '{text_field}' as the text field for embeddings")
    
    # Store embeddings in batches
    batch_size = 50
    
    # Better way to iterate through dataset in batches
    for i in range(0, len(dataset), batch_size):
        # Get a slice of the dataset
        batch = dataset.select(range(i, min(i+batch_size, len(dataset))))
        
        ids = []
        documents = []
        embeddings = []
        
        for j, sample in enumerate(batch):
            doc_id = f"doc_{i+j}"
            text = sample[text_field]
            
            # Handle potential list/dict structures
            if isinstance(text, (list, dict)):
                text = str(text)
            
            # Skip empty texts
            if not text or len(text.strip()) == 0:
                continue
                
            embedding = compute_embedding(text)
            
            ids.append(doc_id)
            documents.append(text)
            embeddings.append(embedding)
        
        if ids:  # Only add if there are valid documents
            collection.add(ids=ids, documents=documents, embeddings=embeddings)
        
        print(f"Stored embeddings for {len(ids)} documents")
    
    print(f"Stored embeddings for {collection.count()} documents")

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

def interactive_query():
    """
    Process queries from the dataset instead of interactive input.
    """
    print("\n=== RAG Query Processing ===")
    print(f"Database contains {collection.count()} documents")
    
    # Load the dataset directly
    dataset = load_data()
    print(f"Loaded {len(dataset)} samples from {DATASET_NAME}")
    
    # Process each query in the dataset using the 'query' field
    for i, sample in enumerate(tqdm(dataset, desc="Processing queries")):
        # Skip if the sample doesn't have a query field
        if 'query' not in sample:
            continue
            
        query = sample['query']
        
        # Skip empty queries
        if not query or len(query.strip()) == 0:
            continue
            
        print(f"\nQuery {i+1}: {query}")
        
        results = query_database(query)
        
        print(f"Top {len(results['documents'][0])} results:")
        for j, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            print(f"\n--- Result {j+1} (Similarity: {1-distance:.4f}) ---")
            # Print a preview of the document (first 200 chars)
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            print(preview)
            
        print("\n" + "-"*50)
        
        # Optional: add a limit to avoid processing too many queries
        if i >= 50:  # Process only the first 50 queries
            print("Reached the maximum number of queries to process.")
            break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="RAG Inference with ChromaDB")
    parser.add_argument("--store", action="store_true", help="Store embeddings in the database")
    parser.add_argument("--query", action="store_true", help="Query the database interactively")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME, 
                        help=f"HuggingFace dataset name (default: {DATASET_NAME})")
    parser.add_argument("--samples", type=int, default=MAX_SAMPLES,
                        help=f"Maximum number of samples to process (default: {MAX_SAMPLES})")
    
    args = parser.parse_args()
    
    # Update global variables based on args
    DATASET_NAME = args.dataset
    MAX_SAMPLES = args.samples
    
    if args.store:
        store_embeddings()
    
    if args.query or (not args.store and not args.query):
        interactive_query()
