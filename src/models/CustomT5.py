import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoTokenizer
from utils.model_utils import reshape_embedding_to_tensors, adapt_embeddings

base_model = "google-t5/t5-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_tokenizer = T5Tokenizer.from_pretrained(base_model)

class CustomT5(T5ForConditionalGeneration):
     """ custom t5 with gnn and projector and coverage loss"""
     def __init__(self, custom_config):
          custom_config.use_cache = False
          super().__init__(custom_config)
          self.gradient_checkpointing_enable()
          
          if not hasattr(custom_config, 'projector_input_size'):
               raise ValueError("CustomT5 requires 'projector_input_size' in config.")

          hidden_size = max(1024, self.config.d_model * 2)
          self.projector = ResProjector(
               custom_config.projector_input_size,
               hidden_size,
               self.config.d_model
          )
          
          self.pro_in = custom_config.projector_input_size
          self._freeze_parameters()
          self.encoder.gradient_checkpointing_enable()
     
     def _freeze_parameters(self):
          for param in self.parameters():
               param.requires_grad = False
               
          for param in self.projector.parameters():
               param.requires_grad = True
               
          # unfreeze last 4 layers
          for layer in self.encoder.block[-4:]:
               for param in layer.parameters():
                    param.requires_grad = True
                    
          for layer in self.decoder.block[-4:]:
               for param in layer.parameters():
                    param.requires_grad = True

     def forward(self, attention_mask=None, inputs_embeds=None, labels=None,
                    combin_embeddings_list=None, label_summaries=None, 
                    cov_lambda=0.01,
                    **kwargs
     ):
          inputs_embeds, attention_mask = self._data_combine(
               combin_embeddings_list, inputs_embeds, attention_mask
          )
          
          if label_summaries is not None:
               tokenized_summaries = t5_tokenizer(
                    label_summaries,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    add_special_tokens=True
               )
               labels = tokenized_summaries.input_ids.to(device)

          outputs = super().forward(
               inputs_embeds=inputs_embeds,
               attention_mask=attention_mask,
               labels=labels,
               output_attentions=True,
               **kwargs
          )
          
          ce_loss = outputs.loss  # token-level cross-entropy

          attn_layers = outputs.decoder_attentions
          last_layer_attn = attn_layers[-1].mean(dim=1)  # [batch, tgt_len, src_len]

          coverage = torch.zeros_like(last_layer_attn[:, 0, :])  # [batch, src_len]
          cov_loss = 0.0
          for t in range(last_layer_attn.size(1)):
               a_t = last_layer_attn[:, t, :]  # [batch, src_len]
               cov_loss += torch.sum(torch.min(a_t, coverage), dim=1).mean()
               coverage = coverage + a_t

          loss = ce_loss + cov_lambda * cov_loss

          outputs.loss = loss
          
          return outputs

     def _data_combine(self, combin_embeddings_list=None, inputs_embeds=None, attention_mask=None):
          if combin_embeddings_list is not None:
               inputs_comb_embeds, masks = adapt_embeddings(
                    batch_token_list=combin_embeddings_list,
                    emb_dim=self.pro_in,
                    device=device
               )
               
               inputs_comb_embeds = inputs_comb_embeds.to(device)
               masks = masks.to(device)

               inputs_embeds = self.projector(inputs_comb_embeds)
               attention_mask = masks
               
          return inputs_embeds, attention_mask


class ResProjector(nn.Module):
     def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
          super().__init__()
          if in_dim == out_dim:
               self.shortcut = nn.Identity()
          else:
               self.shortcut = nn.Linear(in_dim, out_dim)
          
          self.norm1 = nn.LayerNorm(in_dim)
          self.dense1 = nn.Linear(in_dim, hidden_dim)
          self.norm2 = nn.LayerNorm(hidden_dim)
          self.dense2 = nn.Linear(hidden_dim, out_dim)
          self.dropout = nn.Dropout(dropout)
          self.activation = nn.GELU()
          
          self._init_weights()

     def _init_weights(self):
          nn.init.xavier_uniform_(self.dense1.weight)
          nn.init.xavier_uniform_(self.dense2.weight)
          nn.init.normal_(self.dense1.bias, std=1e-6)
          nn.init.normal_(self.dense2.bias, std=1e-6)

     def forward(self, x):
          sc = self.shortcut(x)
          
          x = self.norm1(x)
          x = self.dense1(x)
          x = self.activation(x)
          x = self.norm2(x)
          x = self.dropout(x)
          x = self.dense2(x)
          
          return x + sc