import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import T5Config, T5ForConditionalGeneration

from models.RelHetGraph import RelHetGraph
from models.CustomT5 import CustomT5
from models.TextEncoder import LongTextEncoder, LongTextTokenEncoder

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
          self.special_token_embs = self._create_special_embedding(t5_tokenizer, text_encoder_model)

          self.to(self.device)
          
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
                    token_embed = text_encoder_model.shared(torch.tensor([token_id])) # [1, hidden_size]
                    token_embed = self.special_ln(token_embed)
                    gnn_embed = self.llm2gnn(token_embed)  # [1, out_size]

                    cat = torch.cat([gnn_embed, token_embed], dim=-1)
                    spe_token_emb[token_type] = cat.detach()
          
          return spe_token_emb
