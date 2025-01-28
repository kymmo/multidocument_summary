import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as geo_DataLoader
from torch.utils.data import DataLoader as data_DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from models.RelHetGraph import RelHetGraph
from models.DatasetLoader import SummaryDataset, EvalDataset
from utils.model_utils import freeze_model

large_model = "facebook/bart-large"
small_model = "facebook/bart-base" #for test
bart_tokenizer = BartTokenizer.from_pretrained(small_model)
bart_model = BartForConditionalGeneration.from_pretrained(small_model)

def train_gnn_bart_loss(file_path, hidden_size, out_size, num_heads,sentence_in_size = 768, word_in_size = 768, learning_rate=0.001, num_epochs=20, feat_drop=0.2, attn_drop=0.2, batch_size=32):
     """Trains the HetGNN model using a proxy task."""
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
     train_dataset = SummaryDataset(file_path)
     train_dataloader = geo_DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

     gnn_model = RelHetGraph(hidden_size, out_size, num_heads, sentence_in_size, word_in_size , feat_drop, attn_drop).to(device)
     BART_embed_layer_projector = nn.Linear(out_size, bart_model.config.d_model).to(device) ## size needed: (batch_size, sequence_length, hidden_size)
     optimizer = torch.optim.Adam(list(gnn_model.parameters()) + list(BART_embed_layer_projector.parameters()), lr=learning_rate)
     
     freeze_model(bart_model)
     bart_model.eval() ## no update for model
     gnn_model.train() ## set to train mode
     for epoch in range(num_epochs):
          total_loss = 0
          for batch in train_dataloader:
               batch = batch.to(device)
               sentence_feat = batch['sentence'].x
               word_feat = batch['word'].x
               
               ## adding data noise
               corrupted_sentence_feat = F.dropout(sentence_feat, p=0.1, training=True)

               # forward
               optimizer.zero_grad()
               sentence_embeddings = gnn_model(batch, corrupted_sentence_feat, word_feat)

               # BART NLL loss
               projected_embeddings = BART_embed_layer_projector(sentence_embeddings)
               reshape_embeddings = projected_embeddings.unsqueeze(1)  # fit the BART input need (batch_size, sequence_length, hidden_size)
               
               ## labels calculate
               bart_embedding_matrix = bart_model.get_input_embeddings().weight  # (vocab_size, hidden_dim)
               similarities = chunked_cosine_similarity(projected_embeddings, bart_embedding_matrix, chunk_size=16)
               top_k_values, top_k_indices = similarities.topk(k=5, dim=1)
               average_similarity = top_k_values.mean(dim=1)  # (batch_size,)
               abs_diff = torch.abs(similarities - average_similarity.unsqueeze(1))  # (batch_size, vocab_size)
               closest_token_ids = abs_diff.argmin(dim=1)  # (batch_size,)
               seq_length = reshape_embeddings.size(1)
               labels = closest_token_ids.unsqueeze(1).expand(-1, seq_length)  # (batch_size, seq_length)
               labels = labels.long() # make sure long type

               outputs = bart_model(inputs_embeds=reshape_embeddings, labels=labels)
               loss = outputs.loss ## cross-entropy
               # ## distribution
               # logits = outputs.logits
               # log_probs = F.log_softmax(logits, dim=-1)
               # loss = F.kl_div(log_probs, labels, reduction='batchmean')

               loss.backward()
               optimizer.step()
               total_loss += loss.item()

          print(f"Epoch {epoch+1}/{num_epochs}, Learning rate: {learning_rate}, Loss: {total_loss / len(train_dataloader)}")

     torch.save(gnn_model.state_dict(), 'gnn_trained_weights.pth')
     torch.save(BART_embed_layer_projector.state_dict(), 'bart_projector_weights.pth')
     
     print("GNN Training Finish.")
     
     return gnn_model

def get_gnn_bart_trained_embedding(evl_data_path, hidden_size, out_size, num_heads,sentence_in_size = 768, word_in_size = 768, feat_drop=0.2, attn_drop=0.2, batch_size=32):
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     ## must be the same para of the train model
     gnn_model = RelHetGraph(hidden_size, out_size, num_heads, sentence_in_size, word_in_size , feat_drop, attn_drop).to(device)
     gnn_model.load_state_dict(torch.load('gnn_trained_weights.pth', weights_only=True))
     gnn_model.eval()
     
     evl_dataset = EvalDataset(evl_data_path)
     eval_dataloader = data_DataLoader(evl_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
     
     output_embeddings = []
     node_sent_maps = []
     summary_list = []
     with torch.no_grad():
          for batch in eval_dataloader:
               batch_graph, batch_map, batch_summary = batch
               batch_graph = batch_graph.to(device)
          
               sentence_feat = batch_graph['sentence'].x
               word_feat = batch_graph['word'].x
               
               embeddings = gnn_model(batch_graph, sentence_feat, word_feat)
               output_embeddings.append(embeddings)
               node_sent_maps.append(batch_map)
               summary_list.append(batch_summary)

     output_embeddings = torch.cat(output_embeddings, dim=0)
     merged_node_map_list = [item for sublist in node_sent_maps for item in sublist]
     merged_summary_list = [item for batch_sum in summary_list for item in batch_sum]

     return output_embeddings, merged_node_map_list, merged_summary_list

def chunked_cosine_similarity(embeddings, embedding_matrix, chunk_size=16):
     similarities = []
     for i in range(0, embeddings.size(0), chunk_size):
          chunk = embeddings[i:i + chunk_size]
          sim = F.cosine_similarity(chunk.unsqueeze(1), embedding_matrix.unsqueeze(0), dim=2)
          similarities.append(sim)
     return torch.cat(similarities, dim=0)

def custom_collate_fn(batch):
     graphs, node_maps, summary_list = zip(*batch)

     batched_graph = Batch.from_data_list(graphs)
     batched_maps = []
     batch_summary = []
     for node_map, summary in zip(node_maps, summary_list):
          batched_maps.append(node_map)
          batch_summary.append(summary) ## string list
     
     return batched_graph, batched_maps, batch_summary