from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from huggingface_hub import login

token = ''
login(token=token)

model_name = "meta-llama/Meta-Llama-3-70B"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
