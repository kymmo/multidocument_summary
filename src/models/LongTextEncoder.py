import torch

from utils.model_utils import clean_memory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LongTextEncoder:
          def __init__(self, tokenizer, model, chunk_size = 512, overlap_percent = 0.2):
               self.tokenizer = tokenizer
               self.model = model
               self.chunk_size = chunk_size
               self.overlap_percent = overlap_percent
               
          def encode(self, text):
               tokens = self.tokenizer.tokenize(text)
               if len(tokens) <= self.chunk_size:
                    inputs = self.tokenizer(text, return_tensors="pt", 
                                        max_length=self.chunk_size, 
                                        truncation=True).to(device)
                    return self.model.encoder(**inputs).last_hidden_state.mean(1)
               
               ## long text encode
               chunks = []
               start = 0
               overlap_tokens = int(self.chunk_size * self.overlap_percent)
               step = self.chunk_size  - overlap_tokens
               
               while start < len(tokens):
                    end = min(start + self.chunk_size, len(tokens))
                    chunk = self.tokenizer.convert_tokens_to_string(tokens[start:end])
                    chunks.append(chunk)
                    start += step
               
               chunk_embs = []
               for chunk in chunks:
                    with torch.no_grad(), torch.cuda.amp.autocast():
                         inputs = self.tokenizer(chunk, return_tensors="pt").to(device)
                         emb = self.model.encoder(**inputs).last_hidden_state.mean(1)
                         chunk_embs.append(emb.cpu())
                    del inputs
               
               clean_memory()
               
               return torch.mean(torch.stack(chunk_embs).to(device), dim=0)