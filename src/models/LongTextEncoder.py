import torch

from utils.model_utils import clean_memory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LongTextEncoder:
          def __init__(self, tokenizer, model, chunk_size = 512, overlap_percent = 0.2):
               self.tokenizer = tokenizer
               self.model = model
               self.chunk_size = chunk_size
               self.overlap_percent = overlap_percent
          
          def split_text(self, text):
               tokens = self.tokenizer.tokenize(text)
               if len(tokens) <= self.chunk_size:
                    return [text]
               
               chunks = []
               start = 0
               total_tokens = len(tokens)
               overlap_tokens = int(self.chunk_size * self.overlap_percent)
               
               while start < total_tokens:
                    end = min(start + self.chunk_size, total_tokens)
                    chunk = self.tokenizer.convert_tokens_to_string(tokens[start:end])
                    chunks.append(chunk)
                    if end >= total_tokens:
                         break
                    start = end - overlap_tokens
               
               return chunks
               
          def encode(self, text):
               tokens = self.tokenizer.tokenize(text)
               if len(tokens) <= self.chunk_size:
                    with torch.no_grad(), torch.cuda.amp.autocast():
                         inputs = self.tokenizer(text, return_tensors="pt", 
                                             max_length=self.chunk_size, 
                                             truncation=True).to(device)
                         return self.model.encoder(**inputs).last_hidden_state.mean(1).cpu()
               
               ## long text encode
               chunks = self.split_text(text)
               chunk_embs = []
               for chunk in chunks:
                    with torch.no_grad(), torch.cuda.amp.autocast():
                         inputs = self.tokenizer(
                              chunk,
                              max_length=512,
                              truncation=True,
                              padding='longest',
                              return_tensors='pt'
                         ).to(device)
                         emb = self.model.encoder(**inputs).last_hidden_state.mean(1)
                         chunk_embs.append(emb.cpu())
                    del inputs
               
               clean_memory()
               
               return torch.mean(torch.stack(chunk_embs).to(device), dim=0)
          
          def encode_batch(self, texts, batch_size=16):
               all_chunks = []
               chunk_info = []
               for idx, text in enumerate(texts):
                    chunks = self.split_text(text)
                    all_chunks.extend(chunks)
                    chunk_info.append( (idx, len(chunks)) )
               
               chunk_embs = []
               for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i+batch_size]
                    with torch.no_grad(), torch.cuda.amp.autocast():
                         inputs = self.tokenizer(
                              batch_chunks,
                              max_length=512,
                              truncation=True,
                              padding='longest',
                              return_tensors='pt'
                         ).to(device)
                         
                         outputs = self.model.encoder(**inputs)
                    
                    embs = outputs.last_hidden_state.mean(1).cpu()
                    chunk_embs.append(embs)
                    del inputs, outputs
               
               chunk_embs = torch.cat(chunk_embs)
               text_embs = [[] for _ in range(len(texts))]
               chunk_ptr = 0
               for idx, chunk_size in chunk_info:
                    text_embs[idx] = torch.mean(chunk_embs[chunk_ptr:chunk_ptr + chunk_size], dim=0)
                    chunk_ptr += chunk_size
               
               return torch.stack(text_embs)