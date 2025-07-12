import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from itertools import groupby
import torch
from torch_geometric.nn import GATConv, HeteroConv

class EdgeKeyTuple(Enum):
     SENT_SIM = ('sentence', 'similarity', 'sentence')
     SENT_ANT = ('sentence', 'pro_ant', 'sentence')
     WORD_SENT = ('word', 'in', 'sentence')
     SENT_WORD = ('sentence', 'has', 'word')
     DOC_SENT = ('document', 'has', 'sentence')
     SENT_DOC = ('sentence', 'in', 'document')
          
          
class RelHetGraph(nn.Module):
     
     def __init__(self,  hidden_size, out_size, num_heads, projection_dim, 
                    sentence_in_size = 768, word_in_size = 768, 
                    document_in_size = 768, feat_drop=0.1, attn_drop=0.1):
          super().__init__()
          
          self.lin_sent = nn.Linear(sentence_in_size, sentence_in_size)
          self.lin_word = nn.Linear(word_in_size, word_in_size)
          self.lin_doc = nn.Linear(document_in_size, document_in_size)
          
          # GAT1
          conv1_dict = {
               EdgeKeyTuple.SENT_SIM.value: GATConv(in_channels=sentence_in_size, out_channels=hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.SENT_ANT.value: GATConv(in_channels=sentence_in_size, out_channels=hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               
               EdgeKeyTuple.WORD_SENT.value: GATConv(in_channels=(word_in_size, sentence_in_size), out_channels=hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.SENT_WORD.value: GATConv(in_channels=(sentence_in_size, word_in_size), out_channels=hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               
               EdgeKeyTuple.DOC_SENT.value: GATConv(in_channels=(document_in_size, sentence_in_size), out_channels=hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.SENT_DOC.value: GATConv(in_channels=(sentence_in_size, document_in_size), out_channels=hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
          }
          self.conv1 = HeteroConv(conv1_dict, aggr='sum')
          
          # GAT2
          conv2_input_dim = hidden_size * num_heads
          conv2_dict = {
               EdgeKeyTuple.SENT_SIM.value: GATConv(in_channels=conv2_input_dim, out_channels=out_size, heads=1, concat=False, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.SENT_ANT.value: GATConv(in_channels=conv2_input_dim, out_channels=out_size, heads=1, concat=False, dropout=attn_drop, add_self_loops=False),
               
               EdgeKeyTuple.WORD_SENT.value: GATConv(in_channels=(conv2_input_dim, conv2_input_dim), out_channels=out_size, heads=1, concat=False, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.SENT_WORD.value: GATConv(in_channels=(conv2_input_dim, conv2_input_dim), out_channels=out_size, heads=1, concat=False, dropout=attn_drop, add_self_loops=False),
               
               EdgeKeyTuple.DOC_SENT.value: GATConv(in_channels=(conv2_input_dim, conv2_input_dim), out_channels=out_size, heads=1, concat=False, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.SENT_DOC.value: GATConv(in_channels=(conv2_input_dim, conv2_input_dim), out_channels=out_size, heads=1, concat=False, dropout=attn_drop, add_self_loops=False),
          }
          self.conv2 = HeteroConv(conv2_dict, aggr='sum')
          
          # dropout
          self.feat_drop = nn.Dropout(feat_drop)
          
          self.projection_head_sentence = nn.Sequential(
               nn.Linear(out_size, out_size // 2),
               nn.ReLU(),
               nn.Linear(out_size // 2, projection_dim)
          )

     def forward(self, hetero_data, need_projection = False):
          h_transformed = {}
          
          if 'sentence' in hetero_data.node_types:
               sentence_embeddings = self.lin_sent(hetero_data['sentence'].x)
               h_transformed['sentence'] = F.relu(sentence_embeddings)
               
          if 'word' in hetero_data.node_types:
               word_embeddings = self.lin_word(hetero_data['word'].x)
               h_transformed['word'] = F.relu(word_embeddings)
               
          if 'document' in hetero_data.node_types:
               document_embeddings = self.lin_doc(hetero_data['document'].x)
               h_transformed['document'] = F.relu(document_embeddings)
          
          if not h_transformed:
               print("[Warning] No node features found in x_dict to process in GNN.")
               return None, None
          
          h1_output = self.conv1(h_transformed, hetero_data.edge_index_dict)
          h1_processed = {}
          for node_type, h_val in h1_output.items():
               h_val_activated = F.relu(h_val)
               h_val_dropped = self.feat_drop(h_val_activated)
               h_val_norm = F.layer_norm(h_val_dropped, h_val_dropped.shape[-1:])
               h1_processed[node_type] = h_val_norm.flatten(start_dim=1)
          
          h2_normed = {}
          h2_output = self.conv2(h1_processed, hetero_data.edge_index_dict)
          for ntype, h_val in h2_output.items():
               h2_normed[ntype] = F.layer_norm(h_val, h_val.shape[-1:])
               
          sentence_output_h2 = h2_normed.get('sentence')
          sentence_embeddings_gnn = None # for downstream
          sentence_embeddings_projected = None # for contrastive

          if sentence_output_h2 is not None:
               sentence_embeddings_gnn = sentence_output_h2
               
               if need_projection:
                    sentence_embeddings_projected = self.projection_head_sentence(sentence_output_h2)

          return sentence_embeddings_gnn, sentence_embeddings_projected
     
     def _get_pos_emb(self, batch_label):
          pos_embs = []
          
          labels = [list(group) for key, group in groupby(batch_label)]
          seg_embs = self.segment_embeddings(torch.arange(len(labels)))
          
          for idx, group in enumerate(labels):
               p_emb = self.position_embeddings(torch.arange(len(group)))
               s_emb = seg_embs[idx].unsqueeze(0)
               cur_emb = p_emb + s_emb
               pos_embs.extend(cur_emb)
          
          return torch.stack(pos_embs, dim=0)