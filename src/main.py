# from langchain.llms import HuggingFaceLLM
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load the GovReport dataset
dataset = load_dataset("ccdv/govreport-summarization")

# Step 2: Load the T5 tokenizer and model
cur_model_name = "t5-small"  # Use a larger model like t5-base if needed
pretrained_model = T5ForConditionalGeneration.from_pretrained(cur_model_name)
tokenizer = T5Tokenizer.from_pretrained(cur_model_name)

# Step 3: Preprocess the dataset for fine-tuning
def preprocess_function(examples):
     inputs = tokenizer(examples["document"], padding="max_length", truncation=True, max_length=512)
     labels = tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=150)
     inputs["labels"] = labels["input_ids"]
     return inputs

# Apply preprocessing to the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Split into train and validation datasets
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

# Step 4: Set up the training arguments for fine-tuning
training_args = TrainingArguments(
     output_dir="./results",
     evaluation_strategy="epoch",
     learning_rate=2e-5,
     per_device_train_batch_size=8,
     per_device_eval_batch_size=8,
     num_train_epochs=3,
     weight_decay=0.01,
)

# Step 5: Initialize the Trainer for fine-tuning
trainer = Trainer(
     model=pretrained_model,
     args=training_args,
     train_dataset=train_dataset,
     eval_dataset=val_dataset,
     tokenizer=tokenizer,
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Evaluate the model after fine-tuning
results = trainer.evaluate()
print(f"Evaluation Results: {results}")

# Step 8: Set up LangChain for multi-document summarization with the fine-tuned T5 model
prompt_template = """You are an expert summarizer. You will be given multiple documents, and your task is to summarize them into a concise summary.
Documents:
{documents}

Summary:"""

# Initialize LangChain components
prompt = PromptTemplate(input_variables=["documents"], template=prompt_template)
llm = HuggingFaceLLM(model_name=cur_model_name, model=pretrained_model, tokenizer=tokenizer)
chain = LLMChain(llm=llm, prompt=prompt)

# Step 9: Summarize documents using the fine-tuned model
def summarize_documents(documents):
     combined_documents = " ".join(documents)
     summary = chain.run({"documents": combined_documents})
     return summary

# Step 10: Test summarization with a sample from the GovReport dataset
sample_documents = dataset[:3]["document"]  # Select the first 3 documents for testing
summary = summarize_documents(sample_documents)
print("Summary after fine-tuning:", summary)
