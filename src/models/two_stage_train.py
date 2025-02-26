import torch
import os
from pathlib import Path
import torch.nn as nn
import time
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.cuda.amp import autocast
from torch_geometric.loader import DataLoader as geo_DataLoader
from torch.utils.data import DataLoader as data_DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

from models.RelHetGraph import RelHetGraph
from models.DatasetLoader import EvalDataset, OptimizedDataset, custom_collate_fn
from models.CustomT5 import CustomT5
from models.gnn_train_t5 import train_gnn
from utils.model_utils import freeze_model, clean_memory

base_model = "google-t5/t5-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_tokenizer = T5Tokenizer.from_pretrained(base_model)
t5_model = T5ForConditionalGeneration.from_pretrained(base_model).to(device)
# model_save_path = os.path.join('.', 'saved_models')

def train_gnn_t5(dataset_path, hidden_size, out_size, num_heads=8, learning_rate=0.001, num_epochs=20, feat_drop=0.1, attn_drop=0.1, batch_size=16):
     ## gnn training, t5 freezed
     print(f"Start training GNN. Parameters: hidden_size: {hidden_size}, out_size: {out_size}, attention_heads: {num_heads}")
     train_data_path = os.path.join(dataset_path, "train.jsonl")
     ## path check
     file = Path(train_data_path)
     if not file.exists():
          raise FileNotFoundError(f"File path {train_data_path} is not exist!")
     print(f"Accessing data path: {train_data_path}")
     
     print("Start training gnn...")
     gnn_start_time = time.time()
     #### train gnn, freeze t5
     train_gnn(
          file_path=train_data_path,
          hidden_size=hidden_size,
          out_size=out_size,
          num_heads=num_heads,
          sentence_in_size = 768,
          word_in_size = 768,
          learning_rate=learning_rate,
          num_epochs=num_epochs,
          feat_drop=feat_drop,
          attn_drop=attn_drop,
          batch_size=batch_size,
          save_method='entire_model'
     )
     gnn_end_time = time.time()
     print(f"Finish gnn training, time cost:  {gnn_end_time - gnn_start_time:.4f} s.")
     
     #### train t5, freeze gnn
     print("Start fine-tuning T5...")
     t5_start_time = time.time()
     fine_tune_t5(
          file_path=train_data_path,
          out_size=out_size,
          num_epochs=num_epochs,
          batch_size=batch_size
     )
     t5_end_time = time.time()
     print(f"Finish T5 fine-tune, time cost:  {t5_end_time - t5_start_time:.4f} s.")
     
     print("Two-stage training finish!")

def get_combined_embed(batch_graph_list, gnn_embeddings, sent_text):
     """ concat gnn_embedding and text t5 embeddings
          output batch graph's sentences embedding list
     """
     # graph_ind = batch['sentence'].ptr.numpy()
     concat_embedding_list = []
     start_ind = 0
     for i_th, graph in enumerate(batch_graph_list): # for each embs of graph
          graph_sent_num = graph['sentence'].x.shape[0]
          gnn_sent_embs = gnn_embeddings[start_ind: start_ind + graph_sent_num]
          start_ind = start_ind + graph_sent_num
          graph_sent = sent_text[i_th]
          
          ## t5 sent sembeddings
          with torch.no_grad():
               graph_sent.insert(0, "Generate a summary from documents' embeddings: ") # task prefix
               padding = torch.zeros(1, gnn_sent_embs.shape[1]).to(device) ## padding for same size to sent
               padding_gnn_embeddings = torch.cat([padding, gnn_sent_embs], dim = 0)
          
               with autocast():
                    ## get T5 embeddings # TODO: deal with long text
                    inputs = t5_tokenizer(
                         graph_sent, 
                         return_tensors="pt", 
                         padding='max_length',
                         truncation=True,
                         max_length=512
                    )
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    t5_model.eval()
                    encoder_sent_outputs = t5_model.encoder(
                              input_ids,
                              attention_mask=attention_mask,
                              return_dict=True,
                    )## encode text infor

                    t5_embeddings = encoder_sent_outputs.last_hidden_state
                    # ignore paddding
                    masked_embeddings = t5_embeddings * attention_mask.unsqueeze(-1)
                    avg_t5_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True) ## (sentence_number, embedding)
                    avg_t5_embeddings = avg_t5_embeddings.to(device)

                    ## concatinate GNN and T5 embedding
                    gnn_emb_norm = nn.LayerNorm(gnn_sent_embs.shape[1])(padding_gnn_embeddings)
                    t5_emb_norm = nn.LayerNorm(t5_model.config.hidden_size)(avg_t5_embeddings)
                    
                    combined_embeddings = torch.cat([gnn_emb_norm, t5_emb_norm], dim=1)
                    
                    concat_embedding_list.append(combined_embeddings)
     
     return concat_embedding_list

def chunked_cosine_similarity(embeddings, embedding_matrix, chunk_size=16):
     similarities = []
     embeddings = embeddings.half().contiguous()
     embedding_matrix = embedding_matrix.half().contiguous()
     
     for i in range(0, embeddings.size(0), chunk_size):
          chunk = embeddings[i:i + chunk_size]
          with torch.no_grad():
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

def fine_tune_t5(file_path, out_size, num_epochs = 20, batch_size=16):
     ## data load
     train_dataset = EvalDataset(file_path)
     train_dataloader = data_DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=True,
          # pin_memory=True, ## data has been in GPU while training gnn
          collate_fn=custom_collate_fn
     )
     
     ## models load
     gnn_model = torch.load('gnn_trained_weights.pt')
     gnn_model.eval()
     freeze_model(gnn_model)
     
     config = T5Config.from_pretrained(base_model)
     config.projector_input_size = out_size + t5_model.config.hidden_size
     custom_t5_model = CustomT5(config)
     optimizer = torch.optim.AdamW(
          [
          {"params": custom_t5_model.encoder.block[-2:].parameters(), "lr": 1e-4},
          {"params": custom_t5_model.decoder.block[-2:].parameters(), "lr": 1e-4},
          {"params": custom_t5_model.projector.parameters(), "lr": 1e-3}
          ],
          weight_decay=0.01
     )
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

     print(f"CUDA usage after model loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB has used, remaining {torch.cuda.max_memory_allocated()/1024**3:.2f} GB available.")
     
     torch.cuda.empty_cache()
     scaler = torch.cuda.amp.GradScaler()
     print(f"Setting finish. Start training epoch...")
     for epoch in range(num_epochs):
          custom_t5_model.train()
          total_loss = 0
          
          for batch in train_dataloader:
               batch_graph, batch_map, batch_summary = batch
               optimizer.zero_grad()
               
               with torch.cuda.amp.autocast():
                    batched_graph = Batch.from_data_list(batch_graph).to(device, non_blocking=True)
                    sentence_feat = batched_graph['sentence'].x
                    word_feat = batched_graph['word'].x
                    sent_text = batched_graph['sentence'].text
                    
                    ## adding data noise
                    corrupted_sentence_feat = F.dropout(sentence_feat, p=0.1, training=gnn_model.training)

                    # forward
                    gnn_embeddings = gnn_model(batched_graph, corrupted_sentence_feat, word_feat)
                    concat_embs_list = get_combined_embed(batch_graph, gnn_embeddings, sent_text)
                    
                    outputs = custom_t5_model(combin_embeddings_list = concat_embs_list, label_summaries=batch_summary)
                    loss = outputs.loss
               
               scaler.scale(loss).backward()
               scaler.step(optimizer)
               scaler.update()
               total_loss += loss.item()
          
          print(f"Epoch {epoch+1} / {num_epochs}, Loss: {total_loss/len(train_dataloader):.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
          scheduler.step()
          
     custom_t5_model.save_pretrained("./fine_tuned_t5")


""" Deprecated"""
def train_gnn_by_cat(file_path, hidden_size, out_size, num_heads,sentence_in_size = 768, word_in_size = 768, learning_rate=0.001, num_epochs=20, feat_drop=0.2, attn_drop=0.2, batch_size=16):
     """Trains the HetGNN model using a proxy task."""
     clean_memory()
     print(f"Task runing on {device}")

     print(f"Start loading sample graphs...")
     train_dataset = OptimizedDataset(file_path)
     train_dataloader = geo_DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=True,
          pin_memory=True,
          follow_batch=['sentence', 'word']
     )
     print(f"Dataset load successfully!")
     
     gnn_model = RelHetGraph(hidden_size, out_size, num_heads, sentence_in_size, word_in_size , feat_drop, attn_drop).to(device)
     concat_dim = out_size + t5_model.config.hidden_size ## embedding concatination
     T5_embed_layer_projector = nn.Sequential(
          nn.Linear(concat_dim, t5_model.config.d_model)
          # nn.LayerNorm(t5_model.config.d_model)
     ).to(device)
     optimizer = torch.optim.Adam(list(gnn_model.parameters()) + list(T5_embed_layer_projector.parameters()), lr=learning_rate)
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

     print(f"CUDA usage after model loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB has used, remaining {torch.cuda.max_memory_allocated()/1024**3:.2f} GB available.")
     
     torch.cuda.empty_cache()
     freeze_model(t5_model)
     scaler = torch.cuda.amp.GradScaler()
     print(f"Setting finish. Start training epoch...")
     for epoch in range(num_epochs):
          gnn_model.train() ## set to train mode
          total_loss = 0
          
          for batch in train_dataloader:
               batch = batch.to(device, non_blocking=True)
               optimizer.zero_grad()
               
               with torch.cuda.amp.autocast():
                    sentence_feat = batch['sentence'].x
                    word_feat = batch['word'].x
                    sent_text = batch['sentence'].text
                    
                    ## adding data noise
                    corrupted_sentence_feat = F.dropout(sentence_feat, p=0.1, training=gnn_model.training)

                    # forward
                    gnn_embeddings = gnn_model(batch, corrupted_sentence_feat, word_feat)
                    concat_embs = get_combined_embed(batch, gnn_embeddings, sent_text)
                    concat_emb_tensors = torch.cat(concat_embs, dim = 0)
                    
                    # T5 NLL loss
                    projected_embeddings = T5_embed_layer_projector(concat_emb_tensors)
                    reshape_embeddings = projected_embeddings.unsqueeze(1)  # fit the T5 input need (batch_size, sequence_length, hidden_size)
                    
                    ## labels calculate ##TODO improved
                    with torch.no_grad():
                         t5_embedding_matrix = t5_model.get_input_embeddings().weight  # (vocab_size, hidden_dim)
                         similarities = chunked_cosine_similarity(projected_embeddings, t5_embedding_matrix, chunk_size=8)
                         # top_k_values, top_k_indices = similarities.topk(k=3, dim=1)
                         # average_similarity = top_k_values.mean(dim=1)  # (batch_size,)
                         # abs_diff = torch.abs(similarities - average_similarity.unsqueeze(1))  # (batch_size, vocab_size)
                         # closest_token_ids = abs_diff.argmin(dim=1)  # (batch_size,) top k average similarity
                         closest_token_ids = similarities.argmax(dim=1) ## most similar
                         seq_length = reshape_embeddings.size(1)
                         labels = closest_token_ids.unsqueeze(1).expand(-1, seq_length) # (batch_size, seq_length)
                         labels = labels.long().to(device)  # make sure long type and GPU calculation

                    
                    outputs = t5_model(inputs_embeds=reshape_embeddings, labels=labels)
                    loss = outputs.loss
               
               scaler.scale(loss).backward()
               scaler.step(optimizer)
               scaler.update()
               total_loss += loss.item()
          
          print(f"Epoch {epoch+1} / {num_epochs}, Loss: {total_loss/len(train_dataloader):.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
          scheduler.step()

     torch.save(gnn_model.state_dict(), 'gnn_trained_weights.pth')
     torch.save(T5_embed_layer_projector.state_dict(), 't5_projector_weights.pth')
     
     print("GNN Training Finish.")
     
     del gnn_model
     del T5_embed_layer_projector
     clean_memory()