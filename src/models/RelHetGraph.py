import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv

class RelHetGraph(nn.Module):
     def __init__(self, sentence_in_size, word_in_size, hidden_size, out_size, num_heads, feat_drop=0.2, attn_drop=0.2):
          super(RelHetGraph, self).__init__()
          
          self.lin_sent = nn.Linear(sentence_in_size, sentence_in_size)
          self.lin_word = nn.Linear(word_in_size, word_in_size)
          
          # GAT1
          self.conv1 = HeteroConv({
               ('sentence', 'to', 'sentence'): GATConv(sentence_in_size, hidden_size, heads=num_heads, dropout=attn_drop),
               ('sentence', 'to', 'word'): GATConv(sentence_in_size, hidden_size, heads=num_heads, dropout=attn_drop),
               ('word', 'to', 'sentence'): GATConv(word_in_size, hidden_size, heads=num_heads, dropout=attn_drop)
          })
          
          # GAT2
          self.conv2 = HeteroConv({
               ('sentence', 'to', 'sentence'): GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop),
               ('sentence', 'to', 'word'): GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop),
               ('word', 'to', 'sentence'): GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop)
          })
          
          # dropout
          self.feat_drop = nn.Dropout(feat_drop)

     def forward(self, g, sentence_feat, word_feat):
          h = {
               'sentence': F.relu(self.lin_sent(sentence_feat)),
               'word': F.relu(self.lin_word(word_feat))
          }
          
          h = self.conv1(h, g.edge_index_dict)
          h = {k: self.feat_drop(h_val) for k, h_val in h.items()}  # dropout
          h = {k: h_val.flatten(1) for k, h_val in h.items()}  # flatten the output
          
          h = self.conv2(h, g.edge_index_dict)
          h = {k: h_val.mean(1) for k, h_val in h.items()}  #mean over heads
          
          return h['sentence']