from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

class T5Model:
     def __init__(self, model_name="t5-small"):
          self.tokenizer = T5Tokenizer.from_pretrained(model_name)
          self.model = T5ForConditionalGeneration.from_pretrained(model_name)

     def fine_tune(self, dataset_path, output_dir, num_epochs=3, batch_size=8):
          """Fine-tune the T5 model."""
          dataset = load_dataset('csv', data_files=dataset_path)

          # Tokenize dataset
          def tokenize_function(examples):
               inputs = [f"summarize: {doc}" for doc in examples["input"]]
               outputs = examples["target"]
               model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
               labels = self.tokenizer(outputs, max_length=512, truncation=True, padding="max_length").input_ids
               model_inputs["labels"] = labels
               return model_inputs

          tokenized_dataset = dataset.map(tokenize_function, batched=True)

          # Training arguments
          training_args = TrainingArguments(
               output_dir=output_dir,
               evaluation_strategy="epoch",
               learning_rate=5e-5,
               per_device_train_batch_size=batch_size,
               num_train_epochs=num_epochs,
               weight_decay=0.01,
               save_total_limit=3,
               logging_dir=f"{output_dir}/logs",
               logging_steps=100,
          )

          # Trainer
          trainer = Trainer(
               model=self.model,
               args=training_args,
               train_dataset=tokenized_dataset["train"],
               eval_dataset=tokenized_dataset.get("validation"),
          )

          # Train the model
          trainer.train()
          self.model.save_pretrained(output_dir)
          print(f"Model fine-tuned and saved to {output_dir}")
