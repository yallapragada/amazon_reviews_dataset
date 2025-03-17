import os
import torch
import faiss
import numpy as np
import argparse
import threading
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# FAISS index storage path
FAISS_INDEX_PATH = "./faiss_index.bin"
DOCS_PATH = "./docs.npy"

# Load model for embedding queries
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device_count = torch.cuda.device_count()
print(f"Using {device_count} GPUs")

# Maximum number of queries to process
MAX_QUERIES = 1000
BATCH_SIZE = 100  # Number of queries per batch

# Function to compute embeddings
def compute_embedding(text, device):
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)  # Mean Pooling
    return outputs.cpu().numpy().flatten()

# Function to query FAISS with multithreading and batch processing
def query_faiss_multigpu(dataset, num_samples, n_results=5):
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCS_PATH):
        print("FAISS index or document store not found. Please run `load_faiss.py` first.")
        return []
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    faiss.extract_index_ivf(index).nprobe = 10  # Increase efficiency
    index = faiss.index_cpu_to_all_gpus(index)  # Enable multi-GPU search
    
    documents = np.load(DOCS_PATH, allow_pickle=True)
    results = []
    
    def query_worker(device, batch_queries, results_list, index, n_results):
        batch_embeddings = np.array([compute_embedding(query, device) for query in batch_queries], dtype=np.float32)
        distances, indices = index.search(batch_embeddings, n_results)
        
        for i, query in enumerate(batch_queries):
            retrieved_docs = [documents[idx] for idx in indices[i] if idx < len(documents)]
            results_list.append((query, retrieved_docs))
    
    threads = []
    results_list = []
    
    for batch_start in tqdm(range(0, num_samples, BATCH_SIZE), desc="Processing Query Batches"):
        batch_dataset = dataset.select(range(batch_start, min(batch_start + BATCH_SIZE, num_samples)))
        batch_queries = [sample["query"] for sample in batch_dataset if "query" in sample]
        batch_threads = []
        
        # Distribute batch across available GPUs
        for i in range(device_count):
            sub_batch = batch_queries[i::device_count]  # Distribute queries evenly across GPUs
            thread = threading.Thread(target=query_worker, args=(i, sub_batch, results_list, index, n_results))
            batch_threads.append(thread)
            thread.start()
        
        for thread in batch_threads:
            thread.join()
import os
import torch
import faiss
import numpy as np
import argparse
import threading
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# FAISS index storage path
FAISS_INDEX_PATH = "./faiss_index.bin"
DOCS_PATH = "./docs.npy"

# Load model for embedding queries
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device_count = torch.cuda.device_count()
print(f"Using {device_count} GPUs")

# Maximum number of queries to process
MAX_QUERIES = 1000
BATCH_SIZE = 100  # Number of queries per batch
K = 5

# Function to compute embeddings
def compute_embedding(text, device):
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)  # Mean Pooling
    return outputs.cpu().numpy().flatten()

# Function to query FAISS with multithreading and batch processing
def query_faiss_multigpu(dataset):
    
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCS_PATH):
        print("FAISS index or document store not found. Please run `load_faiss.py` first.")
        return []
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    faiss.extract_index_ivf(index).nprobe = 10  # Increase efficiency
    index = faiss.index_cpu_to_all_gpus(index)  # Enable multi-GPU search
    
    documents = np.load(DOCS_PATH, allow_pickle=True)
    results = []
    
    def query_worker(device, batch_queries, results_list, index):
        batch_embeddings = np.array([compute_embedding(query, device) for query in batch_queries], dtype=np.float32)
        distances, indices = index.search(batch_embeddings, K)
        
        for i, query in enumerate(batch_queries):
            retrieved_docs = [documents[idx] for idx in indices[i] if idx < len(documents)]
            results_list.append((query, retrieved_docs))
    
    threads = []
    results_list = []
    
    for batch_start in tqdm(range(0, MAX_QUERIES, BATCH_SIZE), desc="Processing Query Batches"):
        batch_dataset = dataset.select(range(batch_start, min(batch_start + BATCH_SIZE, num_samples)))
        batch_queries = [sample["query"] for sample in batch_dataset if "query" in sample]
        batch_threads = []
        
        # Distribute batch across available GPUs
        for i in range(device_count):
            sub_batch = batch_queries[i::device_count]  # Distribute queries evenly across GPUs
            thread = threading.Thread(target=query_worker, args=(i, sub_batch, results_list, index, n_results))
            batch_threads.append(thread)
            thread.start()
        
        for thread in batch_threads:
            thread.join()

    for query, retrieved_docs in results_list:
        print(f"\nQuery: {query}")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. {doc}")
        print('-------------------------')

if __name__ == "__main__":
    dataset = load_dataset("ms_marco", "v2.1", split="train")
    query_faiss_multigpu(dataset)
