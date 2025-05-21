import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from torch_geometric.nn import GATConv, HeteroConv

class EdgeKeyTuple(Enum):
     SENT_SIM = ('sentence', 'similarity', 'sentence')
     SENT_ANT = ('sentence', 'pro_ant', 'sentence')
     WORD_SENT = ('word', 'in', 'sentence')
     SENT_WORD = ('sentence', 'has', 'word')
          
          
class RelHetGraph(nn.Module):
     
     def __init__(self,  hidden_size, out_size, num_heads, sentence_in_size = 768, word_in_size = 768, feat_drop=0.1, attn_drop=0.1):
          super().__init__()
          
          self.lin_sent = nn.Linear(sentence_in_size, sentence_in_size)
          self.lin_word = nn.Linear(word_in_size, word_in_size)
          
          # GAT1
          self.conv1 = HeteroConv({
               EdgeKeyTuple.SENT_SIM.value: GATConv(sentence_in_size, hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.SENT_ANT.value: GATConv(sentence_in_size, hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.WORD_SENT.value: GATConv(word_in_size, hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.SENT_WORD.value: GATConv(word_in_size, hidden_size, heads=num_heads, dropout=attn_drop, add_self_loops=False),
          })
          
          # GAT2
          self.conv2 = HeteroConv({
               EdgeKeyTuple.SENT_SIM.value: GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.SENT_ANT.value: GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.WORD_SENT.value: GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop, add_self_loops=False),
               EdgeKeyTuple.SENT_WORD.value: GATConv(hidden_size * num_heads, out_size, heads=1, dropout=attn_drop, add_self_loops=False),
          })
          
          # dropout
          self.feat_drop = nn.Dropout(feat_drop)

     def forward(self, g, sentence_feat, word_feat):
          h_initial = {
               'sentence': F.relu(self.lin_sent(sentence_feat)),
               'word': F.relu(self.lin_word(word_feat))
          }
          
          h = self.conv1(h_initial, g.edge_index_dict)
          h = {k: self.feat_drop(h_val) for k, h_val in h.items()}  # dropout
          
          h = {k: h_val.flatten(1) for k, h_val in h.items()}  # flatten the output
          h = self.conv2(h, g.edge_index_dict)

          return h['sentence']
     