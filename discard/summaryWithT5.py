from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
from datasets import load_metric
import numpy as np
from typing import List, Dict
import torch
from transformers.utils import sentence_splitter

# Helper function to split text into chunks with overlap and sentence boundaries
def split_text_with_overlap(text, max_length, tokenizer, overlap=50):
     sentences = sentence_splitter.split_text(text)
     chunks = []
     current_chunk = ""

     for sentence in sentences:
          if len(tokenizer.tokenize(current_chunk + " " + sentence)) <= max_length:
               if current_chunk:
                    current_chunk += " " + sentence
               else:
                    current_chunk = sentence
          else:
               if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    # Create overlap by starting with a partial overlap with previous chunk
                    overlap_sentences = []
                    tokens = tokenizer.tokenize(chunks[-1])
                    overlap_sentence = []
                    if len(tokens) > overlap:
                         for k in range(overlap):
                              overlap_sentence.append(tokens[-overlap + k])
                         overlap_text = tokenizer.convert_tokens_to_string(overlap_sentence)
                         current_chunk += overlap_text + " " + sentence
                    else:
                         current_chunk += chunks[-1] + " " + sentence
     if current_chunk:
          chunks.append(current_chunk)

     return chunks

# Step 1: Load the alexfabbri/multi_news dataset
dataset = load_dataset("alexfabbri/multi_news")
print(f"Original Dataset Split Set: {dataset}")

# Step 2: Load the T5 tokenizer and model
cur_model_name = "t5-small"
pretrained_model = T5ForConditionalGeneration.from_pretrained(cur_model_name)
pretrained_tokenizer = T5Tokenizer.from_pretrained(cur_model_name)

# Step 3: Preprocess the dataset for fine-tuning (single-document summarization)
def preprocess_function(examples):
     max_input_length = 512
     max_label_length = 150
     inputs = []
     labels = []
     for documents, summary in zip(examples["document"], examples["summary"]):
          for doc in documents:  # Iterate through individual documents
               doc_chunks = split_text_with_overlap(doc, max_input_length, pretrained_tokenizer)
               for doc_chunk in doc_chunks:
                    inputs.append(pretrained_tokenizer(doc_chunk, padding="max_length", truncation=True, max_length=max_input_length)["input_ids"])
                    labels.append(pretrained_tokenizer(summary, padding="max_length", truncation=True, max_length=max_label_length)["input_ids"])
     return {"input_ids": inputs, "labels": labels}


# Apply preprocessing to the dataset for training the single-doc summarizer
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
print(f"Tokenized dataset split set: {tokenized_datasets}")

# Split into train and validation datasets
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

# Step 4: Set up training arguments for single-document summarization
training_args = TrainingArguments(
     output_dir="./results",
     eval_strategy="epoch",
     learning_rate=2e-5,
     per_device_train_batch_size=8,
     per_device_eval_batch_size=8,
     num_train_epochs=1,
     weight_decay=0.01,
)

# Define metric function to calculate metrics (ROUGE, BERTScore)
def compute_metrics(eval_pred):
     predictions, labels = eval_pred
     decoded_preds = pretrained_tokenizer.batch_decode(predictions, skip_special_tokens=True)
     labels = np.where(labels != -100, labels, pretrained_tokenizer.pad_token_id)
     decoded_labels = pretrained_tokenizer.batch_decode(labels, skip_special_tokens=True)
     rouge = load_metric("rouge")
     rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=True, use_stemmer=True)
     bertscore = load_metric("bertscore")
     bertscore_results = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
     bertscore_results_mean = {}
     for k in bertscore_results:
          bertscore_results_mean[f'mean_{k}'] = np.mean(bertscore_results[k])
     return {**rouge_results, **bertscore_results_mean}

# Step 5: Initialize the Trainer for fine-tuning the single-doc summarizer
trainer = Trainer(
     model=pretrained_model,
     args=training_args,
     train_dataset=train_dataset,
     eval_dataset=val_dataset,
     tokenizer=pretrained_tokenizer,
     compute_metrics = compute_metrics
)

# Step 6: Fine-tune the T5 model for single-document summarization
trainer.train()
# Step 7: Evaluate the model for single-document summarization
results = trainer.evaluate()
print(f"Evaluation Results (Single-Doc): {results}")

# Step 8: Function to Summarize a list of Documents using the finetuned T5 model
def summarize_documents(documents: List[str], model, tokenizer, max_input_length=512, max_summary_length=150) -> List[str]:
     summaries = []
     for doc in documents:
          doc_chunks = split_text_with_overlap(doc, max_input_length, tokenizer)
          doc_summaries = []
          for doc_chunk in doc_chunks:
               inputs = tokenizer(doc_chunk, padding="max_length", truncation=True, max_length=max_input_length, return_tensors="pt").to(model.device)
               summary_ids = model.generate(inputs["input_ids"], max_length=max_summary_length)
               summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
               doc_summaries.append(summary)
          summaries.append(" ".join(doc_summaries))
     return summaries

# Step 9: Function to Combine Individual Summaries into a Multi-Document Summary using the T5 model.
def summarize_multiple_summaries(summaries: List[str], model, tokenizer, max_input_length=512, max_summary_length=150)-> str:
     combined_summaries = " ".join(summaries)
     inputs = tokenizer(combined_summaries, padding="max_length", truncation=True, max_length=max_input_length, return_tensors="pt").to(model.device)
     summary_ids = model.generate(inputs["input_ids"], max_length=max_summary_length)
     final_summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
     return final_summary

# Step 10: Hierarchical multi-document summarization workflow on the validation set.
def evaluate_multidoc_summarization(dataset, model, tokenizer):
     all_summaries = []
     all_labels = []
     for example in dataset:
          documents = example["document"]
          reference_summary = example["summary"]
          individual_summaries = summarize_documents(documents, model, tokenizer)
          multi_doc_summary = summarize_multiple_summaries(individual_summaries, model, tokenizer)
          all_summaries.append(multi_doc_summary)
          all_labels.append(reference_summary)
     rouge = load_metric("rouge")
     rouge_results = rouge.compute(predictions=all_summaries, references=all_labels, use_aggregator=True, use_stemmer=True)
     bertscore = load_metric("bertscore")
     bertscore_results = bertscore.compute(predictions=all_summaries, references=all_labels, lang="en")
     bertscore_results_mean = {}
     for k in bertscore_results:
          bertscore_results_mean[f'mean_{k}'] = np.mean(bertscore_results[k])
     return {**rouge_results, **bertscore_results_mean}

# Step 11: Evaluate multi-document summarization
multidoc_results = evaluate_multidoc_summarization(dataset['validation'], trainer.model, pretrained_tokenizer)
print(f"Evaluation Results (Multi-Doc): {multidoc_results}")