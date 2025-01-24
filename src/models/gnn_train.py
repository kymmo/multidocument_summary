import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from models.RelHetGraph import RelHetGraph
from models.SummaryDataset import SummaryDataset

t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")

def train_gnn(file_path, sentence_in_size, word_in_size, hidden_size, out_size, num_heads, learning_rate=0.001, num_epochs=20, feat_drop=0.2, attn_drop=0.2, batch_size=32):
     """Trains the HetGNN model using a proxy task."""

     train_dataset = SummaryDataset(file_path)
     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

     model = RelHetGraph(sentence_in_size, word_in_size, hidden_size, out_size, num_heads, feat_drop, attn_drop).to(device)

     T5_embed_layer_projector = nn.Linear(out_size, t5_model.config.d_model).to(device)

     optimizer = torch.optim.Adam(list(model.parameters()) + list(T5_embed_layer_projector.parameters()), lr=learning_rate)
     
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
     for epoch in range(num_epochs):
          total_loss = 0
          for batch in train_dataloader:
               batch = batch.to(device)
               sentence_feat = batch['sentence'].x
               word_feat = batch['word'].x

               ## adding data noise
               corrupted_sentence_feat = F.dropout(sentence_feat, p=0.01, training=True)

               # forward
               optimizer.zero_grad()
               sentence_embeddings = model(batch, corrupted_sentence_feat, word_feat)

               # T5 NLL loss
               projected_embeddings = T5_embed_layer_projector(sentence_embeddings)
               outputs = t5_model(inputs_embeds  = projected_embeddings)
               loss = outputs.loss

               loss.backward()
               optimizer.step()
               total_loss += loss.item()

          print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

     return model