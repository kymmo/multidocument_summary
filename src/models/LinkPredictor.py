import torch
import torch.nn as nn

class LinkPredictor(nn.Module):
     def __init__(self, gnn_sentence_out_size, hidden_neurons_scale_factor=0.5, dropout_rate=0.1):
          super().__init__()
          input_dim = gnn_sentence_out_size * 2
          hidden_dim = int(gnn_sentence_out_size * hidden_neurons_scale_factor)

          self.mlp = nn.Sequential(
               nn.Linear(input_dim, hidden_dim),
               nn.ReLU(),
               nn.Dropout(dropout_rate),
               nn.Linear(hidden_dim, 1)
          )

     def forward(self, embedding_i, embedding_j):
          combined = torch.cat([embedding_i, embedding_j], dim=-1)
          return self.mlp(combined)