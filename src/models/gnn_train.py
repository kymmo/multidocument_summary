import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from models.RelHetGraph import RelHetGraph
from models.DatasetLoader import SummaryDataset, EvalDataset

base_model = "google-t5/t5-base"
small_model = "google-t5/t5-small" #for test
# t5_tokenizer = T5Tokenizer.from_pretrained(small_model, legacy=False)
t5_tokenizer = T5Tokenizer.from_pretrained(small_model)
t5_model = T5ForConditionalGeneration.from_pretrained(small_model)

def train_gnn(file_path, hidden_size, out_size, num_heads,sentence_in_size = 768, word_in_size = 768, learning_rate=0.001, num_epochs=20, feat_drop=0.2, attn_drop=0.2, batch_size=32):
     """Trains the HetGNN model using a proxy task."""
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
     train_dataset = SummaryDataset(file_path)
     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

     #####################test
     print(f"data size: {train_dataset.__len__}")
     print(f"hidden_size:{hidden_size}, out_size: {out_size}, num_heads: {num_heads}, sentence_in_size: {sentence_in_size}, word_in_size: {word_in_size}, feat_drop: {feat_drop}, attn_drop: {attn_drop}")
     
     model = RelHetGraph(hidden_size, out_size, num_heads, sentence_in_size, word_in_size , feat_drop, attn_drop).to(device)
     T5_embed_layer_projector = nn.Linear(out_size, t5_model.config.d_model).to(device) ## size needed: (batch_size, sequence_length, hidden_size)
     optimizer = torch.optim.Adam(list(model.parameters()) + list(T5_embed_layer_projector.parameters()), lr=learning_rate)
     
     t5_model.eval() ## no update for T5
     model.train() ## set to train mode
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

               ##################### tset
               print(f"batch_embeddings size: {sentence_embeddings.shape}")
               
               # T5 NLL loss
               projected_embeddings = T5_embed_layer_projector(sentence_embeddings)
               reshape_embeddings = projected_embeddings.unsqueeze(1)  # fit the T5 input need (batch_size, sequence_length, hidden_size)
               ##################### test
               print(f"projected_embeddings size: {projected_embeddings.shape}")
               print("reshape_embeddings shape:", reshape_embeddings.shape)  # 应为 [batch_size, sequence_length, hidden_size]
               print("reshape_embeddings:", reshape_embeddings)
               
               ## labels calculate
               t5_embedding_matrix = t5_model.get_input_embeddings().weight  # (vocab_size, hidden_dim)
               similarities = chunked_cosine_similarity(reshape_embeddings, t5_embedding_matrix, chunk_size=16)
               closest_token_ids = similarities.argmax(dim=1)
               labels = closest_token_ids

               ##################### test
               print(f"closest_token_ids: size: {closest_token_ids.shape}, data: {closest_token_ids}")
               
               with torch.no_grad():
                    outputs = t5_model(inputs_embeds=reshape_embeddings, labels=labels)
               loss = outputs.loss

               loss.backward()
               optimizer.step()
               total_loss += loss.item()

          print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

     torch.save(model.state_dict(), 'gnn_trained_weights.pth')
     
     print("Train Finish.")
     
     return model

def get_gnn_trained_embedding(evl_data_path, sentence_in_size, word_in_size, hidden_size, out_size, num_heads, feat_drop=0.2, attn_drop=0.2, batch_size=32):
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     ## must be the same para of the train model
     gnn_model = RelHetGraph(sentence_in_size, word_in_size, hidden_size, out_size, num_heads, feat_drop, attn_drop)
     gnn_model.load_state_dict(torch.load('gnn_trained_weights.pth', weights_only=True))
     gnn_model.eval()
     
     evl_dataset = EvalDataset(evl_data_path)
     eval_dataloader = DataLoader(evl_dataset, batch_size=batch_size, shuffle=False)
     
     output_embeddings = []
     node_sent_maps = []
     with torch.no_grad():
          for batch in eval_dataloader:
               batch = batch.to(device)
               data, node_map = batch
               sentence_feat = data['sentence'].x
               word_feat = data['word'].x
               
               embeddings = gnn_model(data, sentence_feat, word_feat)
               output_embeddings.append(embeddings)
               node_sent_maps.append(node_map)
               
               ##################### test
               print(f"batch output embedding: {embeddings}")
               print(f"batch node_sent_maps: {node_sent_maps}")

     output_embeddings = torch.cat(output_embeddings, dim=0)
     # node_sent_maps = torch.cat(node_sent_maps, dim=0)
     
     ##################### test
     print(f"output embedding: {output_embeddings}")
     print(f"output embedding: {node_sent_maps}")


     return output_embeddings, node_sent_maps

def chunked_cosine_similarity(embeddings, embedding_matrix, chunk_size=16):
     similarities = []
     for i in range(0, embeddings.size(0), chunk_size):
          chunk = embeddings[i:i + chunk_size]  # 分块
          sim = F.cosine_similarity(chunk.unsqueeze(1), embedding_matrix.unsqueeze(0), dim=2)
          similarities.append(sim)
     return torch.cat(similarities, dim=0)