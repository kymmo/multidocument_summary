import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import LongformerTokenizer
import numpy as np
import random

## TODO: Get summary from SDS
'''
Reference: https://huggingface.co/datasets/alexfabbri/multi_news

Dataset Summary:
Multi-News, consists of news articles and human-written summaries of these articles from the site newser.com. 
Each summary is professionally written by editors and includes links to the original articles cited.

There are two features:
document: text of news articles seperated by special token "|||||".
summary: news summary.
'''
class MultiNewsDataset(Dataset):
     def __init__(self, split_set, tokenizer, max_length=512, graph_nodes_num=10, stride = 256):
          self.dataset = load_dataset("alexfabbri/multi_news", split=split_set)
          self.tokenizer = tokenizer
          self.max_length = max_length
          self.graph_nodes_num = graph_nodes_num
          self.stride = stride
          self.sep_token = "|||||"  # document specified separator

     def __len__(self):
          return len(self.dataset)

     def __getitem__(self, idx): ## idx starts from 0
          item = self.dataset[idx]
          document = item["document"]
          summary = item["summary"]

          # Split the document into individual articles
          articles = document.split(self.sep_token)

          # Tokenize each article individually and keep track of the attention mask
          input_ids_list = []
          attention_mask_list = []
          for article in articles:
               inputs = self.tokenizer(
                    article,
                    max_length=self.max_length,
                    padding=False,
                    truncation=False,
                    return_tensors="pt",
               )
               input_ids = inputs["input_ids"].squeeze()
               attention_mask = inputs["attention_mask"].squeeze()
               input_ids_list.append(input_ids)
               attention_mask_list.append(attention_mask)


          # Combine the tokenized articles (using concatenation as an example)
          self.stride = min(self.stride, self.max_length // 3) ## in case of stride > max_length
          
          input_ids = torch.cat(input_ids_list, dim = 0)
          attention_mask = torch.cat(attention_mask_list, dim = 0)
          # Handle truncation with overlap
          segments = []
          segment_masks = []
          for i in range(0, len(input_ids), self.stride):
               segment_ids = input_ids[i:i+self.max_length]
               segment_mask = attention_mask[i:i+self.max_length]
               # pad the segment if the segment is smaller than max_length
               segment_ids = torch.cat([segment_ids, torch.zeros(self.max_length - len(segment_ids), dtype = torch.long)], dim=0) # pad with zero
               segment_mask = torch.cat([segment_mask, torch.zeros(self.max_length - len(segment_mask), dtype = torch.long)], dim=0) # pad with zero
               segments.append(segment_ids)
               segment_masks.append(segment_mask)
          segments = torch.stack(segments)  # (num_segment, max_length)
          segment_masks = torch.stack(segment_masks) # (num_segment, max_length)


          # Tokenize summary
          summary_inputs = self.tokenizer(
               summary,
               max_length=self.max_length,
               padding="max_length",
               truncation=True,
               return_tensors="pt",
          )

          # Simplified Graph (Random nodes + sentence linking)
          # In real cases, you would use IE or other methods to extract real entities
          # Here we just generate random node ids, with some connections to sentences
          num_sentences = len(articles)
          graph_nodes_ids = random.sample(range(0, num_sentences*5), self.graph_nodes_num) # ids of nodes
          graph_adj = np.zeros((self.graph_nodes_num, num_sentences)) # adjacency matrix
          for i, node_id in enumerate(graph_nodes_ids):
               sentence_id = node_id % num_sentences # link to a sentence (can be improved)
               graph_adj[i, sentence_id] = 1

          # Convert to tensors
          graph_nodes_ids = torch.tensor(graph_nodes_ids, dtype = torch.long)
          graph_adj = torch.tensor(graph_adj, dtype = torch.float)
          return {
               "input_ids": segments,  # use segment of input
               "attention_mask": segment_masks, # use segment of input
               "summary_ids": summary_inputs["input_ids"].squeeze(),
               "summary_attention_mask": summary_inputs["attention_mask"].squeeze(),
               "graph_nodes_ids": graph_nodes_ids,
               "graph_adj": graph_adj,
          }

if __name__ == "__main__":
     # Example usage:
     tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
     dataset = MultiNewsDataset(split="train", tokenizer=tokenizer, max_length = 128, stride = 64)
     sample = dataset[0]

     print("Input IDs shape:", sample["input_ids"].shape) # will be [num_segment, max_length]
     print("Attention Mask shape:", sample["attention_mask"].shape) # will be [num_segment, max_length]
     print("Summary IDs shape:", sample["summary_ids"].shape)
     print("Summary Attention Mask shape:", sample["summary_attention_mask"].shape)
     print("Graph Node IDs shape:", sample["graph_nodes_ids"].shape)
     print("Graph Adj shape:", sample["graph_adj"].shape)