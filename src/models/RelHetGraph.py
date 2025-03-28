import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.transforms import NormalizeFeatures
from torch.utils.checkpoint import checkpoint

class RelHetGraph(nn.Module):
     def __init__(self,  hidden_size, out_size, num_heads, sentence_in_size = 768, word_in_size = 768, feat_drop=0.1, attn_drop=0.1):
          super().__init__()
          
          self.lin_sent = nn.Linear(sentence_in_size, sentence_in_size)
          self.lin_word = nn.Linear(word_in_size, word_in_size)
          
          # GAT1
          self.conv1 = HeteroConv({
               ('sentence', 'similarity', 'sentence'): GATConv(sentence_in_size, hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               ('sentence', 'pro_ant', 'word'): GATConv(sentence_in_size, hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               ('word', 'in', 'sentence'): GATConv(word_in_size, hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               ('sentence', 'has', 'word'): GATConv(word_in_size, hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
          })
          
          # GAT2
          self.conv2 = HeteroConv({
               ('sentence', 'similarity', 'sentence'): GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop, add_self_loops=False),
               ('sentence', 'pro_ant', 'word'): GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop, add_self_loops=False),
               ('word', 'in', 'sentence'): GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop, add_self_loops=False),
               ('sentence', 'has', 'word'): GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop, add_self_loops=False),
          })
          
          # dropout
          self.feat_drop = nn.Dropout(feat_drop)

     def forward(self, g, sentence_feat, word_feat):
          h_initial = {
               'sentence': F.relu(self.lin_sent(sentence_feat)),
               'word': F.relu(self.lin_word(word_feat))
          }
          
          # h = checkpoint(self._forward_conv1, h_initial, g.edge_index_dict, preserve_rng_state=True)
          # h = checkpoint(self._forward_conv2, h, g.edge_index_dict, preserve_rng_state=True)

          h = self.conv1(h_initial, g.edge_index_dict)
          h = {k: self.feat_drop(h_val) for k, h_val in h.items()}  # dropout
          
          h = {k: h_val.flatten(1) for k, h_val in h.items()}  # flatten the output
          h = self.conv2(h, g.edge_index_dict)

          return h['sentence']
     
     def _forward_conv1(self, h_initial, edge_index_dict):
          h = self.conv1(h_initial, edge_index_dict)
          # h = {k: h_val + self.res_proj[k](h_initial[k]) for k, h_val in h.items()}  # residential connect
          h = {k: self.feat_drop(h_val) for k, h_val in h.items()}  # dropout
          
          return h

     def _forward_conv2(self, h, edge_index_dict):
          h = {k: h_val.flatten(1) for k, h_val in h.items()}  # flatten the output
          h = self.conv2(h, edge_index_dict)
          
          return h