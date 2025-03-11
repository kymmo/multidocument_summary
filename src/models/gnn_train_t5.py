import torch
import torch.nn as nn
from torch_geometric.data import Batch
import torch.nn.functional as F
from GPUtil import showUtilization
from torch_geometric.loader import DataLoader as geo_DataLoader
from torch.utils.data import DataLoader as data_DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from models.RelHetGraph import RelHetGraph
from models.DatasetLoader import EvalDataset, OptimizedDataset, custom_collate_fn
from utils.model_utils import freeze_model, clean_memory

base_model = "google-t5/t5-base"
small_model = "google-t5/t5-small" #for test
# t5_tokenizer = T5Tokenizer.from_pretrained(small_model, legacy=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_tokenizer = T5Tokenizer.from_pretrained(base_model)
t5_model = T5ForConditionalGeneration.from_pretrained(base_model).to(device)

def train_gnn(file_path, hidden_size, out_size, num_heads,sentence_in_size = 768, word_in_size = 768, learning_rate=0.001, num_epochs=20, feat_drop=0.2, attn_drop=0.2, batch_size=32, save_method = 'weights'):
     """Trains the HetGNN model using a proxy task."""
     clean_memory()
     print(f"Task runing on {device}")

     print(f"Start loading sample graphs...")
     train_dataset = OptimizedDataset(file_path)
     train_dataloader = geo_DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=True,
          pin_memory=True
          # prefetch_factor=2,
          # num_workers=2,
          )
     print(f"Dataset load successfully!")
     
     projector_hidden_size = 1024
     gnn_model = RelHetGraph(hidden_size, out_size, num_heads, sentence_in_size, word_in_size , feat_drop, attn_drop).to(device)
     T5_embed_layer_projector = nn.Sequential(
               nn.Linear(out_size, projector_hidden_size),
               nn.LayerNorm(projector_hidden_size),
               nn.ReLU(),
               nn.Linear(projector_hidden_size, t5_model.config.d_model)
     ).to(device) ## non_linear transfer
     optimizer = torch.optim.Adam(list(gnn_model.parameters()) + list(T5_embed_layer_projector.parameters()), lr=learning_rate)
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

     # print(f"CUDA usage after model loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB has used, remaining {torch.cuda.max_memory_allocated()/1024**3:.2f} GB available.")
     
     freeze_model(t5_model)
     t5_model.eval() ## no update for T5
     scaler = torch.cuda.amp.GradScaler()
     print(f"Setting finish. Start training epoch...")
     for epoch in range(num_epochs):
          gnn_model.train() ## set to train mode
          total_loss = 0
          for batch in train_dataloader:
               batch = batch.to(device)
               
               with torch.cuda.amp.autocast():
                    sentence_feat = batch['sentence'].x
                    word_feat = batch['word'].x
                    
                    ## adding data noise
                    corrupted_sentence_feat = F.dropout(sentence_feat, p=0.1, training=gnn_model.training)

                    # forward
                    optimizer.zero_grad()
                    sentence_embeddings = gnn_model(batch, corrupted_sentence_feat, word_feat)

                    # T5 NLL loss
                    projected_embeddings = T5_embed_layer_projector(sentence_embeddings)
                    reshape_embeddings = projected_embeddings.unsqueeze(1)  # fit the T5 input need (batch_size, sequence_length, hidden_size)
                    
                    ## labels calculate
                    with torch.no_grad():
                         t5_embedding_matrix = t5_model.get_input_embeddings().weight  # (vocab_size, hidden_dim)
                         similarities = chunked_cosine_similarity(projected_embeddings, t5_embedding_matrix, chunk_size=8)
                         # top_k_values, top_k_indices = similarities.topk(k=5, dim=1)
                         # average_similarity = top_k_values.mean(dim=1)  # (batch_size,)
                         # abs_diff = torch.abs(similarities - average_similarity.unsqueeze(1))  # (batch_size, vocab_size)
                         # closest_token_ids = abs_diff.argmin(dim=1)  # (batch_size,)
                         closest_token_ids = similarities.argmax(dim=1) ## most similar
                         seq_length = reshape_embeddings.size(1)
                         labels = closest_token_ids.unsqueeze(1).expand(-1, seq_length)  # (batch_size, seq_length)
                         labels = labels.long().to(device)  # make sure long type and GPU calculation

                    outputs = t5_model(inputs_embeds=reshape_embeddings, labels=labels)
                    loss = outputs.loss

               scaler.scale(loss).backward()
               scaler.step(optimizer)
               scaler.update()
               total_loss += loss.item()
          
          showUtilization()
          print(f"Epoch {epoch+1} / {num_epochs}, Loss: {total_loss/len(train_dataloader):.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
          scheduler.step()
     
     if save_method == 'entire_model':
          ## save entire model
          torch.save(gnn_model, 'gnn_trained_weights.pt')
     else:
          ## capabal to old version
          torch.save(gnn_model.state_dict(), 'gnn_trained_weights.pth')
          torch.save(T5_embed_layer_projector.state_dict(), 't5_projector_weights.pth')
     
     print("GNN Training Finish.")
     
     del gnn_model
     del T5_embed_layer_projector
     clean_memory()
     

def get_gnn_trained_embedding(evl_data_path, hidden_size, out_size, num_heads,sentence_in_size = 768, word_in_size = 768, feat_drop=0.2, attn_drop=0.2, batch_size=32):
     ## must be the same para of the train model
     torch.cuda.empty_cache()
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
               batch_graph_list, batch_map_list, batch_summary_list = batch

               batch_graph = Batch.from_data_list(batch_graph_list).to(device)
               sentence_feat = batch_graph['sentence'].x
               word_feat = batch_graph['word'].x
               
               embeddings = gnn_model(batch_graph, sentence_feat, word_feat)
               output_embeddings.append(embeddings)
               node_sent_maps.append(batch_map_list)
               summary_list.append(batch_summary_list)
               
               del batch_graph
               clean_memory()

     output_embeddings = torch.cat(output_embeddings, dim=0)
     merged_node_map_list = [item for sublist in node_sent_maps for item in sublist]
     merged_summary_list = [item for batch_sum in summary_list for item in batch_sum]

     return output_embeddings, merged_node_map_list, merged_summary_list

def chunked_cosine_similarity(embeddings, embedding_matrix, chunk_size=16):
     similarities = []
     for i in range(0, embeddings.size(0), chunk_size):
          chunk = embeddings[i:i + chunk_size]
          with torch.no_grad():
               chunk = chunk.half().contiguous()
               embedding_matrix = embedding_matrix.half().contiguous()
               sim = F.cosine_similarity(
                    chunk.unsqueeze(1),
                    embedding_matrix.unsqueeze(0),
                    dim=2
               )
               sim = sim.float()
               
          similarities.append(sim.cpu())
          
          del chunk  # save gpu memory
          del sim
          torch.cuda.empty_cache()
          
     return torch.cat(similarities, dim=0)