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
          if seq_len <= self.target_len:
               # Pad if the input is already shorter than the target length
               padding_size = self.target_len - seq_len
               padding = torch.zeros(batch_size, padding_size, x.shape[2], device=x.device)
               x = torch.cat([x, padding], dim=1)
               mask = torch.cat([torch.ones(batch_size, seq_len), torch.zeros(batch_size, padding_size)], dim=1)
               
               return x, mask.to(x.device)

          weights = self.importance_predictor(x).squeeze(-1)
          weights = F.softmax(weights, dim=-1)
          positions = torch.linspace(0, 1, seq_len, device=x.device)
          position_sim = -torch.abs(self.pool_positions.to(x.device)[None, :, None] - positions[None, None, :]) * seq_len
          combined = position_sim + weights[:, None, :] * 10
          window_centers = torch.softmax(combined, dim=-1)
          compressed = torch.einsum('bts,bsd->btd', window_centers, x)
          
          mask = torch.ones(batch_size, self.target_len, device=x.device)
          
          return compressed, mask