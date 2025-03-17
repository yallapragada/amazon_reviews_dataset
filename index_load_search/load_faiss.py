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

# Maximum number of documents to process
MAX_SAMPLES = 100000
BATCH_SIZE = 50  # Number of documents per batch

def compute_embedding(text, device):
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)  # Mean Pooling
    return outputs.cpu().numpy().flatten()

def embedding_worker(device, batch_texts, index, documents, lock):
    batch_embeddings = []
    batch_documents = []
    
    for text in batch_texts:
        text = str(text).strip()
        if not text:
            continue
        embedding = compute_embedding(text, device)
        batch_embeddings.append(embedding)
        batch_documents.append(text)
    
    if batch_embeddings:
        batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
        
        with lock:
            index.add(batch_embeddings)
            documents.extend(batch_documents)
            faiss.write_index(index, FAISS_INDEX_PATH)
            np.save(DOCS_PATH, np.array(documents, dtype=object))

# Function to store embeddings in FAISS using multiple GPUs and threads
def store_embeddings():
    dataset = load_dataset("ms_marco", "v2.1", split="train")
    num_samples = min(MAX_SAMPLES, len(dataset))
    print(f"Loaded {num_samples} samples from ms_marco")
    
    if os.path.exists(DOCS_PATH):
        documents = list(np.load(DOCS_PATH, allow_pickle=True))
    else:
        documents = []
    
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        index = None
    
    lock = threading.Lock()
    
    for batch_start in tqdm(range(0, num_samples, BATCH_SIZE), desc="Processing Document Batches"):
        batch_dataset = dataset.select(range(batch_start, min(batch_start + BATCH_SIZE, num_samples)))
        batch_texts = [sample["passages"]["passage_text"] for sample in batch_dataset if "passages" in sample]
        
        threads = []
        
        for i in range(device_count):
            sub_batch = batch_texts[i::device_count]  # Distribute texts across GPUs
            thread = threading.Thread(target=embedding_worker, args=(i, sub_batch, index, documents, lock))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        print(f"Added batch {batch_start // BATCH_SIZE + 1} with {len(batch_texts)} documents.")
    
    print(f"Stored {len(documents)} documents in FAISS index.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=MAX_SAMPLES, help="Maximum number of samples")
    args = parser.parse_args()
    
    MAX_SAMPLES = args.samples
    
    store_embeddings()
