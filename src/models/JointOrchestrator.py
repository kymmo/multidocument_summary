import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

from models.RelHetGraph import RelHetGraph
from models.CustomT5 import CustomT5
from models.LongTextEncoder import LongTextEncoder

class JointOrchestrator(nn.Module):
     """
     The master model that orchestrates the entire joint training pipeline.
     It contains the GNN, Text Encoder, and CustomT5 as submodules and
     implements the full data flow within its forward pass.
     """
     def __init__(self, gnn_config: dict, t5_config: T5Config, text_encoder_model: T5ForConditionalGeneration, t5_tokenizer):
          super().__init__()
          
          self.gnn = RelHetGraph(**gnn_config)
          self.text_encoder = LongTextEncoder(t5_tokenizer, text_encoder_model)
          
          doc_bias_size = 30
          t5_config.projector_input_size = gnn_config['out_size'] + text_encoder_model.config.hidden_size + doc_bias_size
          self.custom_t5 = CustomT5(t5_config)
          
          self.global_u = nn.Parameter(torch.randn(gnn_config['out_size']))
          self.global_v = nn.Parameter(torch.randn(gnn_config['out_size']))
          
          with torch.no_grad():
               self.global_u.data = self.global_u / self.global_u.norm()
               self.global_v.data = self.global_v - torch.dot(self.global_v, self.global_u) * self.global_u
               self.global_v.data = self.global_v / self.global_v.norm()
     
     def _data_process(self, batched_graph, graph_list):
          with torch.no_grad():
               sentence_graph_embs, _ = self.gnn(batched_graph)
               sentence_graph_embs = sentence_graph_embs.detach()
          
          sent_texts = batched_graph['sentence'].text
          sent_text_list = [sent for doc in sent_texts for sent in doc]
          prompt = "Summarize: "
          sent_text_list.insert(0, prompt)
          
          ##############test
          print("prompt_sent list: ", sent_text_list)
          ########################
          sentence_text_embs = self.text_encoder.encode_batch(sent_text_list)

          combin_embeddings_list = self._combine_embeddings(
               graph_list,
               sentence_graph_embs,
               sentence_text_embs
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
          
          torch.manual_seed(24)
          doc_discrimination_strength = 1.5
          max_doc_id = 100
          
          for i_th, graph in enumerate(batch_graph_list):
               graph_sent_num = graph['sentence'].x.shape[0]
               gnn_sent_embs = sentence_graph_embs[start_ind: start_ind + graph_sent_num]
               start_ind += graph_sent_num
               t5_sent_embs = sentence_text_embs[text_start_index: text_start_index + graph_sent_num]
               text_start_index += graph_sent_num
               
               gnn_mean = gnn_sent_embs.mean(dim=0)
               gnn_std = gnn_sent_embs.std(dim=0) + 1e-8
               gnn_sent_embs = (gnn_sent_embs - gnn_mean) / gnn_std
               
               gnn_prompt_emb = torch.zeros_like(gnn_sent_embs[0].unsqueeze(0))  # (1, dim)
               text_prompt_emb = sentence_text_embs[0].unsqueeze(0)  # (1, dim)
               
               doc_id = i_th % max_doc_id
               angle = doc_id * 2 * torch.pi / (max_doc_id / doc_discrimination_strength)
               
               u_component = torch.einsum('bi,i->b', gnn_sent_embs, self.global_u).unsqueeze(1) * self.global_u.unsqueeze(0)
               v_component = torch.einsum('bi,i->b', gnn_sent_embs, self.global_v).unsqueeze(1) * self.global_v.unsqueeze(0)
               other_component = gnn_sent_embs - u_component - v_component
               
               rotation_strength = 0.4 + 0.8 * (doc_id % 5) / 4  # [0.4, 1.2]
               
               cos_angle = torch.cos(torch.tensor(angle * rotation_strength))
               sin_angle = torch.sin(torch.tensor(angle * rotation_strength))
               
               gnn_rotated = (
                    cos_angle * u_component +
                    sin_angle * v_component +
                    other_component
               )
               
               gnn_combined = torch.cat([
                    gnn_prompt_emb,
                    gnn_rotated
               ], dim=0)
               
               t5_mean = t5_sent_embs.mean(dim=0)
               t5_std = t5_sent_embs.std(dim=0) + 1e-8
               t5_sent_embs = (t5_sent_embs - t5_mean) / t5_std
               
               text_combined = torch.cat([
                    text_prompt_emb,
                    t5_sent_embs
               ], dim=0)
               
               doc_bias = torch.zeros(30, device=gnn_sent_embs.device)
               doc_bias[:10] = (doc_id % 10) / 80.0
               if doc_id % 10 < 10:
                    doc_bias[10 + doc_id % 10] = 0.08
               if doc_id // 10 < 10:
                    doc_bias[20 + doc_id // 10] = 0.08
               # part1 = torch.ones(10, device=gnn_sent_embs.device) * (doc_id % 10) / 80.0
               # part2 = torch.eye(10, device=gnn_sent_embs.device)[doc_id % 10].float() * 0.08
               # part3 = torch.eye(10, device=gnn_sent_embs.device)[doc_id // 10].float() * 0.08
               # doc_bias = torch.cat([part1, part2, part3])
               
               if doc_bias.dim() == 1:
                    doc_bias = doc_bias.unsqueeze(0)
               
               doc_bias = doc_bias.repeat(gnn_combined.shape[0], 1)
               
               combined = torch.cat([
                    gnn_combined,
                    text_combined,
                    doc_bias
               ], dim=-1)
               
               layer_norm = torch.nn.LayerNorm(combined.shape[-1], elementwise_affine=False)
               combined = layer_norm(combined)
               
               concat_embedding_list.append(combined)
               
          return concat_embedding_list