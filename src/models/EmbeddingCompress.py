import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePoolCompressor(nn.Module):
     def __init__(self, emb_dim, target_len=512):
          super().__init__()
          self.target_len = target_len
          
          self.importance_predictor = nn.Sequential(
               nn.Linear(emb_dim, 256),
               nn.GELU(),
               nn.Linear(256, 1)
          )
          
          self.pool_positions = nn.Parameter(torch.linspace(0, 1, target_len))
          
     def forward(self, x):
          batch_size, seq_len, _ = x.shape
          
          weights = self.importance_predictor(x).squeeze(-1)  # [batch, seq_len]
          weights = F.softmax(weights, dim=-1)
          
          positions = torch.linspace(0, 1, seq_len, device=x.device)  # [seq_len]
          
          # [batch, target_len, seq_len] = [1, target_len, 1] vs [1, 1, seq_len]
          position_sim = -torch.abs(
               self.pool_positions[None, :, None] - positions[None, None, :]
          ) * seq_len
          
          combined = position_sim + weights[:, None, :] * 10
          
          window_centers = torch.softmax(combined, dim=-1)  # [batch, target_len, seq_len]
          
          compressed = torch.einsum('bts,bsd->btd', window_centers, x)
          
          return compressed