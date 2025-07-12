import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import T5Config, T5Tokenizer, T5EncoderModel

from models.EmbeddingCompress import AdaptivePoolCompressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LongTextEncoder(nn.Module):
     def __init__(self, tokenizer, model, chunk_size=512, overlap_percent=0.2):
          super().__init__()
          
          self.tokenizer = tokenizer
          self.model = model
          self.model.to(device)
          self.chunk_size = chunk_size
          self.stride = int(chunk_size * (1 - overlap_percent))
          
          hidden_size = self.model.config.hidden_size
          self.attn_pool = EnhancedAttnPool(hidden_size)
          self.dropout = nn.Dropout(0.2)
     
     def forward(self, texts: list[str], max_subbatch_chunks: int = 200) -> torch.Tensor:
          encoding = self.tokenizer(
               texts,
               max_length=self.chunk_size,
               truncation='longest_first',
               padding='longest',
               return_overflowing_tokens=True,
               return_tensors="pt",
               stride=self.stride
          )
          input_ids = encoding["input_ids"]            # (total_chunks, chunk_size)
          attention_mask = encoding["attention_mask"]  # (total_chunks, chunk_size)
          sample_map = encoding["overflow_to_sample_mapping"]

          total_chunks = input_ids.size(0)
          hidden_dim = self.model.config.hidden_size
          num_texts = len(texts)

          chunk_embs_list = []
          chunks_processed = 0
          while chunks_processed < total_chunks:
               end_idx = min(chunks_processed + max_subbatch_chunks, total_chunks)
               sub_input_ids = input_ids[chunks_processed:end_idx].to(device)
               sub_attention_mask = attention_mask[chunks_processed:end_idx].to(device)

               encoder_outputs = self.model.encoder(
                    input_ids=sub_input_ids,
                    attention_mask=sub_attention_mask
               )
               hidden_states = encoder_outputs.last_hidden_state
               hidden_states = self.dropout(hidden_states)
               pooled = self.attn_pool(hidden_states, sub_attention_mask)
               
               chunk_embs_list.append(pooled)
                    
               chunks_processed = end_idx

          chunk_embs = torch.cat(chunk_embs_list, dim=0)  # (total_chunks, hidden)
          text_embs = torch.zeros((num_texts, hidden_dim), device=device)
          counts = torch.zeros((num_texts,), device=device)
          sample_map_tensor = torch.as_tensor(sample_map, dtype=torch.long, device=device)

          for idx_chunk in range(total_chunks):
               doc_idx = sample_map_tensor[idx_chunk].item()
               text_embs[doc_idx] += chunk_embs[idx_chunk]
               counts[doc_idx] += 1
          counts = counts.clamp(min=1).unsqueeze(1)
          text_embs = text_embs / counts  # (num_texts, hidden)
          
          return text_embs

class EnhancedAttnPool(nn.Module):
     def __init__(self, hidden_size):
          super().__init__()
          self.query = nn.Linear(hidden_size, hidden_size)
          self.key = nn.Linear(hidden_size, hidden_size)
          self.hidden_size = hidden_size

     def forward(self, hidden_states, mask):
          Q = self.query(hidden_states.mean(dim=1, keepdim=True))  # [B, 1, D]
          K = self.key(hidden_states)  # [B, L, D]
          
          scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
          scores = scores.squeeze(1)  # [B, L]
          scores = scores.masked_fill(mask == 0, -1e9)
          
          alphas = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
          
          return (alphas * hidden_states).sum(dim=1)  # [B, D]
     
     
class LongTextTokenEncoder(nn.Module):
     def __init__(self, tokenizer, model, chunk_size=512, overlap_percent=0.2):
          super().__init__()
          self.tokenizer = tokenizer
          self.model = model
          self.model.to(device)
          self.chunk_size = chunk_size
          self.stride = int(chunk_size * (1 - overlap_percent))
     
     def forward(self, texts: list[str], max_subbatch_chunks: int = 200) -> list[torch.Tensor]:
          all_token_embeddings = []
          
          for text in texts:
               if not text.strip():
                    dummy_embed = torch.zeros(1, self.model.config.hidden_size, device=device)
                    all_token_embeddings.append(dummy_embed)
                    continue
                    
               encoding = self.tokenizer(
                    text,
                    max_length=self.chunk_size,
                    truncation=True,
                    padding='max_length',
                    return_overflowing_tokens=True,
                    return_tensors="pt",
                    stride=self.stride,
                    add_special_tokens=True
               )
               
               input_ids = encoding["input_ids"]
               attention_mask = encoding["attention_mask"]
               total_chunks = input_ids.size(0)
               
               token_embeddings = []
               
               chunks_processed = 0
               while chunks_processed < total_chunks:
                    end_idx = min(chunks_processed + max_subbatch_chunks, total_chunks)
                    sub_input_ids = input_ids[chunks_processed:end_idx].to(device)
                    sub_attention_mask = attention_mask[chunks_processed:end_idx].to(device)
                    
                    with torch.no_grad():
                         encoder_outputs = self.model.encoder(
                              input_ids=sub_input_ids,
                              attention_mask=sub_attention_mask
                         )
                    
                    hidden_states = encoder_outputs.last_hidden_state
                    
                    for i in range(sub_input_ids.size(0)):
                         actual_length = sub_attention_mask[i].sum().item()
                         token_embeddings.append(hidden_states[i, :actual_length])
                    
                    chunks_processed = end_idx
               
               if token_embeddings:
                    full_embeddings = torch.cat(token_embeddings, dim=0)
                    all_token_embeddings.append(full_embeddings)
               else:
                    print(f"[WARNING] token embedding is empty.")
          
          return all_token_embeddings
     
class PrefixEncoder(nn.Module):
     """
     Encodes the GNN embeddings into a prefix for the T5 model.
     """
     def __init__(self, config: T5Config, gnn_out_size: int, prefix_length: int):
          super().__init__()
          self.prefix_length = prefix_length
          self.prefix_dim = config.d_model
          
          self.transform = nn.Sequential(
               nn.Linear(gnn_out_size, self.prefix_dim * self.prefix_length),
          )

     def forward(self, sentence_gnn_embeddings: torch.Tensor) -> torch.Tensor:
          # aggregate sentence embeddings to a single document-cluster representation
          doc_cluster_embedding = torch.mean(sentence_gnn_embeddings, dim=0, keepdim=True)
          
          prefix_flat = self.transform(doc_cluster_embedding)
          # Reshape to (batch_size=1, prefix_length, d_model)
          prefix = prefix_flat.view(-1, self.prefix_length, self.prefix_dim)
          return prefix
     
class LongTextEncoderEnhanced(nn.Module):
     """
     Encodes text that is longer than the model's max input size.
     It does this by chunking the text, getting embeddings for each chunk,
     and then using a compressor to create a single fixed-size representation.
     """
     def __init__(self, t5_model_name: str, tokenizer: T5Tokenizer, target_len: int = 512, chunk_size: int = 400, stride: int = 200):
          super().__init__()
          self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          self.encoder = T5EncoderModel.from_pretrained(t5_model_name).to(self.device)
          self.encoder.eval()
          for param in self.encoder.parameters():
               param.requires_grad = False
               
          self.tokenizer = tokenizer
          self.compressor = AdaptivePoolCompressor(self.encoder.config.d_model, target_len)
          self.chunk_size = chunk_size
          self.stride = stride

     def forward(self, long_text_list: List[str]) -> torch.Tensor:
          """ long_text_list: list of docs_text of one sample
          """
          batch_embeddings = []
          for text in long_text_list:
               # 1. Tokenize the entire long text
               token_ids = self.tokenizer.encode(text, add_special_tokens=False)
               
               # 2. Create overlapping chunks
               chunks = [token_ids[i:i + self.chunk_size] for i in range(0, len(token_ids), self.stride)]
               if not chunks:
                    chunks = [[self.tokenizer.pad_token_id]]
               
               chunk_tensors = self.tokenizer.pad(
                    {"input_ids": chunks},
                    padding="longest",
                    return_tensors="pt"
               )['input_ids'].to(self.device)

               # 3. Get embeddings for each chunk
               with torch.no_grad():
                    chunk_embeddings = self.encoder(input_ids=chunk_tensors).last_hidden_state

               # 4. Concatenate embeddings from all chunks to form one long sequence
               full_embedding = torch.cat([emb.view(-1, emb.shape[-1]) for emb in chunk_embeddings], dim=0)
               batch_embeddings.append(full_embedding)
          
          max_len = max(emb.shape[0] for emb in batch_embeddings)
          padded_batch = torch.stack([
               F.pad(emb, (0, 0, 0, max_len - emb.shape[0])) for emb in batch_embeddings
          ])
          
          compressed_embeddings, attention_mask = self.compressor(padded_batch)
          
          return compressed_embeddings, attention_mask