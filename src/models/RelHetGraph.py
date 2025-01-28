import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv

class RelHetGraph(nn.Module):
     def __init__(self,  hidden_size, out_size, num_heads, sentence_in_size = 768, word_in_size = 768, feat_drop=0.2, attn_drop=0.2):
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
          h = self.conv1(h_initial, g.edge_index_dict)
          # h = {k: h_val + h_initial[k] for k, h_val in h.items()}  # residential connect
          h = {k: self.feat_drop(h_val) for k, h_val in h.items()}  # dropout
          
          h = {k: h_val.flatten(1) for k, h_val in h.items()}  # flatten the output
          h = self.conv2(h, g.edge_index_dict)
          
          return h['sentence']