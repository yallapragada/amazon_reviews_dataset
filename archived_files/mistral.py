import numpy as np
import torch
from datasets import load_dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, MistralForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding


raw_dataset = load_dataset("csv", data_files="Gift_Cards.csv", delimiter=",")


def preprocess(example):
    example["label"] = int(float(example["rating"]))
    example["input_text"] = f"Title: {example['title']}\nReview: {example['text']}"
    return example

processed_dataset = raw_dataset.map(preprocess)

split_dataset = processed_dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
})

unique_labels = sorted(list(set(dataset["train"]["label"])))
num_labels = len(unique_labels)

id2label = {i: str(label) for i, label in enumerate(unique_labels)}
label2id = {str(label): i for i, label in enumerate(unique_labels)}

model_checkpoint = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = MistralForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    # Depending on your GPU memory you might use device_map="auto"
    device_map="auto"
)

def tokenize_function(examples):
    # Tokenize the combined input text; adjust max_length as needed.
    return tokenizer(examples["input_text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 6. Data collator (dynamically pads examples to the longest in the batch)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./mistral-finetune",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,    # Adjust batch size based on GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

trainer.save_model("./mistral-finetuned-classifier")
tokenizer.save_pretrained("./mistral-finetuned-classifier")
