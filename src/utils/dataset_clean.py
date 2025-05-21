### for record. actual running process is in Colab

from datasets import load_dataset
import json
import os

def convert_wcep_file(input_file, output_file):
     MAX_LEN = 1000000
     with open(input_file, "r", encoding="utf-8") as fin, \
          open(output_file, "w", encoding="utf-8") as fout:

          for line in fin:
               # ignore empty line
               if not line.strip():
                    continue

               try:
                    data = json.loads(line)
               
                    # check if data exist
                    if "document" in data and isinstance(data["document"], list):
                         # convert to: ["a","d","f"] → [["a"],["d"],["f"]]
                         data["document"] = [[item] for item in data["document"] if len(item) < MAX_LEN]
                    
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
               
               except json.JSONDecodeError:
                    print(f"Ignore unvalid data line:  {line}")

def multinews_clean(save_path):
     dataset = load_dataset("alexfabbri/multi_news")
     train_data = dataset["train"].map(preprocess_function2, batched=True)
     val_data = dataset["validation"].map(preprocess_function2, batched=True)
     test_data = dataset["test"].map(preprocess_function2, batched=True)

     save_to_jsonl(train_data, os.path.join(save_path, "train.jsonl"))
     save_to_jsonl(val_data, os.path.join(save_path, "validation.jsonl"))
     save_to_jsonl(test_data, os.path.join(save_path, "test.jsonl"))
     
def save_to_jsonl(data, filename):
     with open(filename, "w", encoding="utf-8") as f:
          for sample in data:
               json_line = json.dumps(sample, ensure_ascii=False)
               f.write(json_line + "\n")

import re

def preprocess_function2(samples):
     MAX_LEN = 1000000
     
     if 'document' not in samples or not isinstance(samples['document'], list):
          raise ValueError("Invalid 'document' field in samples.")

     # split by '|||||'
     doc_list = []
     for doc in samples['document']:
          if not isinstance(doc, str):
               raise ValueError("Each 'document' item must be a string.")

          parts = [
               re.sub(
                    r'\.+', '.',
                    re.sub(r'\s+', ' ',
                         part.strip()
                         .replace('\n', '.')
                         .replace('\r', '.')
                    )
                    .replace(' .', '.')
                    .replace('. ', '.')
               ).strip('.')
               for part in doc.split('|||||')
               if part.strip()
          ]
          doc_list.append(parts)

     # doc to list
     converted_list = [[
          [item]
          for item in sublist if len(item) < MAX_LEN and len(item) > 0]
                         for sublist in doc_list]

     
     if 'summary' not in samples or not isinstance(samples['summary'], list):
          raise ValueError("Invalid 'summary' field in samples.")

     summary_list = [summary[1:].strip() for summary in samples['summary']] ##去除 '-'

     return {"document": converted_list, "summary": summary_list}