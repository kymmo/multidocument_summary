import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import os
import json
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from models.RelHetGraph import RelHetGraph
from models.CustomT5 import CustomT5, CustomT5WithPrefix
from models.CustomEncoder import LongTextEncoder, LongTextTokenEncoder, PrefixEncoder, LongTextEncoderEnhanced

torch.autograd.set_detect_anomaly(True)

class JointOrchestrator(nn.Module):
     """
     The model that orchestrates the entire joint training pipeline.
     It contains the GNN, Text Encoder, and CustomT5 as submodules and
     implements the full data flow within its forward pass.
     """
     def __init__(self, gnn_config: dict, t5_config: T5Config, text_encoder_model: T5ForConditionalGeneration, t5_tokenizer):
          super().__init__()
          
          self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

          self.gnn = RelHetGraph(**gnn_config)
          # self.text_encoder = LongTextEncoder(t5_tokenizer, text_encoder_model)
          self.text_token_encoder = LongTextTokenEncoder(t5_tokenizer, text_encoder_model)
          
          doc_bias_size = 0
          t5_config.projector_input_size = gnn_config['out_size'] + text_encoder_model.config.hidden_size + doc_bias_size
          self.custom_t5 = CustomT5(t5_config)
          
          self.llm2gnn = nn.Linear(text_encoder_model.config.hidden_size, gnn_config["out_size"])
          nn.init.xavier_uniform_(self.llm2gnn.weight)
          nn.init.constant_(self.llm2gnn.bias, 0.0)
          
          self.special_ln = nn.LayerNorm(text_encoder_model.config.hidden_size)
          
          self.to(self.device)
          self.special_token_embs = self._create_special_embedding(t5_tokenizer, text_encoder_model)
          
     def _data_process(self, batched_graph, graph_list):
          sentence_graph_embs, _ = self.gnn(batched_graph)
          sent_texts = batched_graph['sentence'].text
          sent_text_list = [sent for doc in sent_texts for sent in doc]
          prompt = "Summarize: "
          sent_text_list.insert(0, prompt)
          sentence_token_embs = self.text_token_encoder(sent_text_list)

          combin_embeddings_list = self._combine_token_embs(
               graph_list,
               sentence_graph_embs,
               sentence_token_embs
          )
          
          return combin_embeddings_list
          
     def forward(self, batched_graph, label_summaries, graph_list,  **kwargs):
          combin_embeddings_list = self._data_process(batched_graph, graph_list)
          
          return self.custom_t5(
               combin_embeddings_list=combin_embeddings_list,
               label_summaries=label_summaries,
               **kwargs
          )
     
     def _combine_embeddings(self, batch_graph_list, sentence_graph_embs, sentence_text_embs):
          concat_embedding_list = []
          start_ind = 0
          text_start_index = 1
          
          for i_th, graph in enumerate(batch_graph_list):
               graph_sent_num = graph['sentence'].x.shape[0]
               gnn_sent_embs = sentence_graph_embs[start_ind: start_ind + graph_sent_num]
               start_ind += graph_sent_num
               t5_sent_embs = sentence_text_embs[text_start_index: text_start_index + graph_sent_num]
               text_start_index += graph_sent_num
               
               txt_prompt = sentence_text_embs[0].unsqueeze(0)
               gnn_prompt = self.llm2gnn(txt_prompt)
               
               gnn_combined = torch.cat([gnn_prompt, gnn_sent_embs], dim=0)  # [n+1, g_dim]
               gnn_combined = F.normalize(gnn_combined, p=2, dim=1)
               text_combined = torch.cat([txt_prompt, t5_sent_embs], dim=0)
               
               combined = torch.cat([
                    gnn_combined,
                    text_combined
               ], dim=-1)
               
               # combined = F.layer_norm(combined, combined.shape[-1:])
               concat_embedding_list.append(combined)
               
          return concat_embedding_list
     
     def _combine_token_embs(self, batch_graph_list, sentence_graph_embs: torch.Tensor, token_embs_list: List[torch.Tensor]):
          concat_embedding_list = []
          start_ind = 0
          text_start_index = 1
          
          txt_prompt = token_embs_list[0].mean(dim=0, keepdim=True)
          promp_token_emb = token_embs_list[0]
          gnn_prompt = self.llm2gnn(txt_prompt)
          
          for graph in batch_graph_list:
               graph_sent_num = graph['sentence'].x.shape[0]
               gnn_sent_embs = sentence_graph_embs[start_ind: start_ind + graph_sent_num]
               start_ind += graph_sent_num
               sent_token_list = token_embs_list[text_start_index: text_start_index + graph_sent_num]
               text_start_index += graph_sent_num
               
               gnn_combined = torch.cat([gnn_prompt, gnn_sent_embs], dim=0)  # [n+1, g_dim]
               gnn_combined = F.normalize(gnn_combined, p=2, dim=1)
               
               sent_token_list.insert(0, promp_token_emb)
               
               graph_tokens = []
               for ith, sent_token_embs in enumerate(sent_token_list):
                    corr_gnn_emb = gnn_combined[ith]
                    corr_gnn_emb = F.normalize(corr_gnn_emb, p=2, dim=0)
                    
                    combined_token_embs = []
                    for tok_emb in sent_token_embs:
                         tok_emb = F.normalize(tok_emb, p=2, dim=0)
                         
                         combined = torch.cat([
                              tok_emb.unsqueeze(0),
                              corr_gnn_emb.unsqueeze(0)
                         ], dim=-1)
                         
                         combined_token_embs.append(combined)
                         
                    if ith > 0: ## not prompt
                         combined_token_embs.append(self.special_token_embs['EOS'])
                    
                    graph_tokens.append(torch.stack(combined_token_embs))
               
               concat_embedding_list.append(graph_tokens)
               
          return concat_embedding_list
     
     def _create_special_embedding(self, t5_tokenizer, text_encoder_model):
          special_tokens = {
               "EOS": "</s>"
          }
          spe_token_emb = {}

          with torch.no_grad():
               for token_type, token in special_tokens.items():
                    token_id = t5_tokenizer.convert_tokens_to_ids(token)
                    token_embed = text_encoder_model.shared(torch.tensor([token_id], device=self.device)) # [1, hidden_size]
                    token_embed = self.special_ln(token_embed)
                    gnn_embed = self.llm2gnn(token_embed)  # [1, out_size]

                    cat = torch.cat([gnn_embed, token_embed], dim=-1)
                    spe_token_emb[token_type] = cat.detach()
          
          return spe_token_emb
     
     
class JointOrchestratorwithPrefix(nn.Module):
     """
     Orchestrates the joint training pipeline with a long text encoder and prefix-tuning.
     """
     def __init__(self, gnn_config: dict, t5_model_name: str, prefix_length: int, t5_tokenizer):
          super().__init__()
          self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

          self.gnn = RelHetGraph(**gnn_config)
          
          t5_config = T5Config.from_pretrained(t5_model_name)
          self.custom_t5 = CustomT5WithPrefix(t5_config)
          self.prefix_encoder = PrefixEncoder(t5_config, gnn_config['out_size'], prefix_length)
          
          self.long_text_encoder = LongTextEncoderEnhanced(t5_model_name=t5_model_name, tokenizer=t5_tokenizer,
                                                            chunk_size=400, target_len=512, stride=400)
          
          self.tokenizer = t5_tokenizer
          
          self.config = {
               "gnn_config": gnn_config,
               "t5_model_name": t5_model_name,
               "prefix_length": prefix_length
          }
          
          self._freeze_parameters()
          self.to(self.device)

     def _freeze_parameters(self):
          # Freeze
          for param in self.parameters():
               param.requires_grad = False
                              
          # Unfreeze the trainable components
          for param in self.gnn.parameters(): 
               param.requires_grad = True
          
          for param in self.prefix_encoder.parameters(): 
               param.requires_grad = True
               
          # Unfreeze top layers of T5
          for layer in self.custom_t5.encoder.block[-2:]:
               for param in layer.parameters():
                    param.requires_grad = True
          for layer in self.custom_t5.decoder.block[-2:]:
               for param in layer.parameters():
                    param.requires_grad = True

     def forward(self, source_text_list: List[str], batched_graph, label_summaries: List[str], **kwargs):
          """
          Args:
               source_text_list (List[str]): A list of long concatenated source documents string.
               batched_graph (torch_geometric.data.HeteroData): The batched graph.
               label_summaries (List[str]): The list of target summary strings.
          """
          batched_graph = batched_graph.to(self.device)
          
          source_embeds, source_mask = self.long_text_encoder(source_text_list)
          source_embeds = source_embeds.to(self.device)
          source_embeds.requires_grad_(True)
          source_mask = source_mask.to(self.device)
          
          sentence_graph_embs, _ = self.gnn(batched_graph)
          prefix_embeds = self.prefix_encoder(sentence_graph_embs, batched_graph['sentence'].batch)
          
          labels = self.tokenizer(
               label_summaries, return_tensors="pt", padding=True, truncation=True
          ).input_ids.to(self.device)

          outputs = self.custom_t5(
               inputs_embeds=source_embeds,
               attention_mask=source_mask,
               prefix_embeds=prefix_embeds,
               labels=labels,
               **kwargs
          )

          return outputs

     @torch.no_grad()
     def generate(self, source_text_list: List[str], batched_graph, **kwargs):
          self.eval()
          
          batched_graph = batched_graph.to(self.device)
          
          source_embeds, source_mask = self.long_text_encoder(source_text_list)
          source_embeds = source_embeds.to(self.device)
          source_mask = source_mask.to(self.device)
          
          sentence_graph_embs, _ = self.gnn(batched_graph)
          prefix_embeds = self.prefix_encoder(sentence_graph_embs, batched_graph['sentence'].batch)
          
          full_input_embeds = torch.cat([prefix_embeds.expand(source_embeds.shape[0], -1, -1), source_embeds], dim=1)
          full_attention_mask = torch.cat([
               torch.ones(prefix_embeds.shape[0], prefix_embeds.shape[1], device=self.device).expand(source_mask.shape[0], -1), 
               source_mask
          ], dim=1)
          
          generated_ids = self.custom_t5.generate(
               inputs_embeds=full_input_embeds,
               attention_mask=full_attention_mask,
               **kwargs
          )
          
          return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)