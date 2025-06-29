import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoTokenizer

base_model = "google-t5/t5-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_tokenizer = T5Tokenizer.from_pretrained(base_model)
auto_tokenizer = AutoTokenizer.from_pretrained(base_model)

class CustomT5(T5ForConditionalGeneration):
     """ custom t5 with gnn and projector"""
     def __init__(self, custom_config):
          custom_config.use_cache = False
          super().__init__(custom_config)
          self.gradient_checkpointing_enable()
          
          if not hasattr(custom_config, 'projector_input_size'):
               raise ValueError("CustomT5 requires 'projector_input_size' in config.")

          hidden_size = max(1024, self.config.d_model * 2)
          self.projector = nn.Sequential(
               nn.Linear(custom_config.projector_input_size, hidden_size),
               nn.GELU(),
               nn.LayerNorm(hidden_size),
               nn.Linear(hidden_size, self.config.d_model),
               nn.Tanh(),
          )
          
          self._freeze_parameters()
          self.encoder.gradient_checkpointing_enable()
     
     def _freeze_parameters(self):
          for param in self.parameters():
               param.requires_grad = False
          
          for param in self.projector.parameters():
               param.requires_grad = True
               
          # unfreeze to learn
          for layer in self.encoder.block[-4:]:
               for param in layer.parameters():
                    param.requires_grad = True
                    
          for layer in self.decoder.block[-4:]:
               for param in layer.parameters():
                    param.requires_grad = True

     def forward(self, attention_mask=None, inputs_embeds=None, labels=None, combin_embeddings_list=None, label_summaries=None, **kwargs):
          if combin_embeddings_list is not None:
               inputs_comb_embeds, masks = reshape_embedding_to_tensors(combin_embeddings_list) # [batch_size, seq_len, embedding_size]
               inputs_embeds = self.projector(inputs_comb_embeds).to(device) # [batch, seq_len, d_model]
               attention_mask = masks.to(device)
          
          if label_summaries is not None:
               tokenized_summaries = t5_tokenizer(
                    label_summaries, # string list
                    padding="max_length",
                    truncation = True,
                    max_length=512,
                    return_tensors="pt",
                    add_special_tokens=True
               )
               labels = tokenized_summaries.input_ids.to(device)
          
          return super().forward(
               inputs_embeds=inputs_embeds,
               attention_mask=attention_mask,
               labels=labels,
               **kwargs
          )
          
     
def reshape_embedding_to_tensors(combin_embeddings_list, max_len = 512):
     processed_embeddings_list = [emb.to(device) for emb in combin_embeddings_list]
     
     reshape_list = [] ##[tensor(1, sequence_size, embed_size)]
     masks = [] ## (batch_size, sequence_size)
     
     max_node_num = max(graph_embs.shape[0] for graph_embs in processed_embeddings_list)
     max_node_num = min(max_len, max_node_num)
     for graph_embs in processed_embeddings_list:
          cur_len = graph_embs.shape[0]
          
          padding_size = max_node_num - cur_len
          if padding_size > 0:
               graph_embs = torch.cat([
                    graph_embs,
                    torch.zeros(padding_size, graph_embs.shape[1], device=device)
               ], dim=0)
          elif padding_size < 0:
               embs_unsqueezed = graph_embs.unsqueeze(0)
               pooled_embs = F.adaptive_avg_pool1d(embs_unsqueezed.transpose(1, 2), max_len)
               graph_embs = pooled_embs.transpose(1, 2).squeeze(0)
               
          reshape_list.append(graph_embs)
          
          mask = torch.zeros(max_node_num, device=device)
          mask_len = cur_len if cur_len <= max_len else max_len
          mask[:mask_len] = 1
          masks.append(mask)
          
     return torch.stack(reshape_list), torch.stack(masks)
