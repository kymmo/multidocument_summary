# from langchain.llms import HuggingFaceLLM
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
from datasets import load_metric
import numpy as np
from typing import Dict, List

# Step 1: Load the GovReport dataset
dataset = load_dataset("ccdv/govreport-summarization")

print(f"Original Dataset Split Set: {dataset}")  

# Step 2: Load the T5 tokenizer and model
cur_model_name = "t5-base"
pretrained_model = T5ForConditionalGeneration.from_pretrained(cur_model_name)
pretrained_tokenizer = T5Tokenizer.from_pretrained(cur_model_name)

# Step 3: Preprocess the dataset for fine-tuning
def preprocess_function(examples):
     inputs = pretrained_tokenizer(examples["report"], padding="max_length", truncation=True, max_length=512)
     labels = pretrained_tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=150)
     inputs["labels"] = labels["input_ids"]
     return inputs

# Apply preprocessing to the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

print(f"Tokenized dataset split set: {tokenized_datasets}")  

# Split into train and validation datasets
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

# Step 4: Set up the training arguments for fine-tuning
training_args = TrainingArguments(
     output_dir="./results",
     eval_strategy="epoch",
     learning_rate=2e-5,
     per_device_train_batch_size=8,
     per_device_eval_batch_size=8,
     num_train_epochs=3,
     weight_decay=0.01,
)


# Define the metric function to calculate the metrics
def compute_metrics_ROUGE_BERTS(eval_pred):
     predictions, labels = eval_pred
     decoded_preds = pretrained_tokenizer.batch_decode(predictions, skip_special_tokens=True)
     # Replace -100 in the labels as we can't decode them.
     labels = np.where(labels != -100, labels, pretrained_tokenizer.pad_token_id)
     decoded_labels = pretrained_tokenizer.batch_decode(labels, skip_special_tokens=True)

     # Rouge metrics
     rouge = load_metric("rouge")
     rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=True, use_stemmer=True)

     # BERTScore
     bertscore = load_metric("bertscore")
     bertscore_results = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
     bertscore_results_mean = {}
     for k in bertscore_results:
          bertscore_results_mean[f'mean_{k}'] = np.mean(bertscore_results[k])

    return {**rouge_results, **bertscore_results_mean}


# Step 5: Initialize the Trainer for fine-tuning
trainer = Trainer(
     model=pretrained_model,
     args=training_args,
     train_dataset=train_dataset,
     eval_dataset=val_dataset,
     tokenizer=pretrained_tokenizer,
     compute_metrics = compute_metrics_ROUGE_BERTS
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Evaluate the model after fine-tuning
results = trainer.evaluate()
print(f"Evaluation Results: {results}")