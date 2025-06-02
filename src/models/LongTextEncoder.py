import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LongTextEncoder:
     def __init__(self, tokenizer, model, chunk_size=512, overlap_percent=0.2):
          self.tokenizer = tokenizer
          self.model = model
          self.model.eval()
          self.chunk_size = chunk_size
          self.stride = int(chunk_size * (1 - overlap_percent))
          # overlap = chunk_size - stride

     def encode_batch(self, texts: list[str], max_subbatch_chunks: int = 200) -> torch.Tensor:
          encoding = self.tokenizer(
               texts,
               max_length=self.chunk_size,
               truncation=True,
               padding='longest',
               return_overflowing_tokens=True,
               return_tensors="pt",
               stride=self.stride
          )
          input_ids = encoding["input_ids"]            # shape = (total_chunks, chunk_size)
          attention_mask = encoding["attention_mask"]  # shape = (total_chunks, chunk_size)
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

               with torch.no_grad(), torch.cuda.amp.autocast():
                    encoder_outputs = self.model.encoder(
                         input_ids=sub_input_ids,
                         attention_mask=sub_attention_mask
                    )
                    # (sub_num_chunks, chunk_size, hidden_dim)
                    sub_hidden_states = encoder_outputs.last_hidden_state
                    
                    mask = sub_attention_mask.unsqueeze(-1).float()        # (sub_num_chunks, chunk_size, 1)
                    masked_hidden = sub_hidden_states * mask               # (sub_num_chunks, chunk_size, hidden_dim)
                    sum_hidden = masked_hidden.sum(dim=1)                  # (sub_num_chunks, hidden_dim)
                    token_counts = mask.sum(dim=1).clamp(min=1)            # (sub_num_chunks, 1)
                    sub_chunk_embs = sum_hidden / token_counts             # (sub_num_chunks, hidden_dim)
                    
                    chunk_embs_list.append(sub_chunk_embs)
               chunks_processed = end_idx

          chunk_embs = torch.cat(chunk_embs_list, dim=0)
          text_embs = torch.zeros((num_texts, hidden_dim), device=device)
          counts = torch.zeros((num_texts,), device=device)

          sample_map_tensor = torch.as_tensor(sample_map, dtype=torch.long, device=device)
          for idx_chunk in range(total_chunks):
               sample_idx = sample_map_tensor[idx_chunk].item()
               text_embs[sample_idx] += chunk_embs[idx_chunk]
               counts[sample_idx] += 1

          counts = counts.clamp(min=1).unsqueeze(1)
          text_embs = text_embs / counts             # shape = (num_texts, hidden_dim)

          return text_embs