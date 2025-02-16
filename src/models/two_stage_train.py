import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch_geometric.loader import DataLoader as geo_DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from models.RelHetGraph import RelHetGraph
from models.DatasetLoader import EvalDataset, OptimizedDataset
from utils.model_utils import freeze_model, clean_memory

base_model = "google-t5/t5-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_tokenizer = T5Tokenizer.from_pretrained(base_model)
t5_model = T5ForConditionalGeneration.from_pretrained(base_model).to(device)

def train_gnn_t5():
     ##TODO
     #### train gnn, freeze t5
     #### train t5, freeze gnn
     return

def train_gnn_by_cat(file_path, hidden_size, out_size, num_heads,sentence_in_size = 768, word_in_size = 768, learning_rate=0.001, num_epochs=20, feat_drop=0.2, attn_drop=0.2, batch_size=32):
     """Trains the HetGNN model using a proxy task."""
     clean_memory()
     print(f"Task runing on {device}")

     print(f"Start loading sample graphs...")
     train_dataset = OptimizedDataset(file_path)
     train_dataloader = geo_DataLoader(
          train_dataset,
          batch_size=8,
          shuffle=False,
          pin_memory=True,
          follow_batch=['sentence', 'word']
     )
     print(f"Dataset load successfully!")
     
     gnn_model = RelHetGraph(hidden_size, out_size, num_heads, sentence_in_size, word_in_size , feat_drop, attn_drop).to(device)
     concat_dim = out_size + t5_model.config.hidden_size ## embedding concatination
     T5_embed_layer_projector = nn.Sequential(
          nn.Linear(concat_dim, t5_model.config.d_model),
          nn.LayerNorm(t5_model.config.d_model)
     ).to(device)
     optimizer = torch.optim.Adam(list(gnn_model.parameters()) + list(T5_embed_layer_projector.parameters()), lr=learning_rate)
     
     print(f"CUDA usage after model loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB has used, remaining {torch.cuda.max_memory_allocated()/1024**3:.2f} GB available.")
     
     freeze_model(t5_model)
     t5_model.eval() ## no update for T5
     gnn_model.train() ## set to train mode
     print(f"Setting finish. Start training epoch...")
     for epoch in range(num_epochs):
          total_loss = 0
          for batch in train_dataloader:
               batch = batch.to(device)
                    
               sentence_feat = batch['sentence'].x
               word_feat = batch['word'].x
               sent_text = batch['sentence'].text
               
               ## adding data noise
               corrupted_sentence_feat = F.dropout(sentence_feat, p=0.1, training=True)

               # forward
               optimizer.zero_grad()
               gnn_embeddings = gnn_model(batch, corrupted_sentence_feat, word_feat)
               concat_embs = get_combined_embed(batch, gnn_embeddings, sent_text)
               
               # T5 NLL loss
               projected_embeddings = T5_embed_layer_projector(concat_embs)
               reshape_embeddings = projected_embeddings.unsqueeze(1)  # fit the T5 input need (batch_size, sequence_length, hidden_size)
               
               ## labels calculate
               t5_embedding_matrix = t5_model.get_input_embeddings().weight  # (vocab_size, hidden_dim)
               similarities = chunked_cosine_similarity(projected_embeddings, t5_embedding_matrix, chunk_size=8)
               top_k_values, top_k_indices = similarities.topk(k=5, dim=1)
               average_similarity = top_k_values.mean(dim=1)  # (batch_size,)
               abs_diff = torch.abs(similarities - average_similarity.unsqueeze(1))  # (batch_size, vocab_size)
               closest_token_ids = abs_diff.argmin(dim=1)  # (batch_size,)
               seq_length = reshape_embeddings.size(1)
               labels = closest_token_ids.unsqueeze(1).expand(-1, seq_length)  # (batch_size, seq_length)
               labels = labels.long().to(device)  # make sure long type and GPU calculation

               outputs = t5_model(inputs_embeds=reshape_embeddings, labels=labels)
               loss = outputs.loss
               loss.backward()
               optimizer.step()
               total_loss += loss.item()
               
               ## del local variables
               del batch
               del labels
               clean_memory()
               
          print(f"Epoch {epoch+1}/{num_epochs}, Learning rate: {learning_rate}, Loss: {total_loss / len(train_dataloader)}")

     torch.save(gnn_model.state_dict(), 'gnn_trained_weights.pth')
     torch.save(T5_embed_layer_projector.state_dict(), 't5_projector_weights.pth')
     
     print("GNN Training Finish.")
     
     del gnn_model
     del T5_embed_layer_projector
     clean_memory()

def get_combined_embed(batch, gnn_embeddings, sent_text):
     """ concat gnn_embedding and text t5 embeddings
          output batch graph's sentences embedding list
     """
     graph_ind = batch['sentence'].ptr.numpy()
     concat_embedding_list = []
     for i_th, start_ind in enumerate(graph_ind): # for each embs of graph
          if i_th + 1 >= len(graph_ind): break
          
          end_ind = graph_ind[i_th + 1]
          gnn_sent_embs = gnn_embeddings[start_ind:end_ind]
          graph_sent = sent_text[i_th]
          
          ## t5 sent sembeddings
          with torch.no_grad():
               graph_sent.insert(0, "Generate a summary from documents' embeddings: ") # task prefix
               padding = torch.zeros(1, gnn_sent_embs.shape[1]) ## padding for same size to sent
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
                    combined_embeddings = torch.cat([padding_gnn_embeddings, avg_t5_embeddings], dim=1)
                    
                    concat_embedding_list.append(combined_embeddings)
     
     concat_emb_tensors = torch.cat(concat_embedding_list, dim = 0)
     return concat_emb_tensors

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

## TODO:
def fine_tune_t5(gnn_sent_embeddings, sample_node_sent_maps, sequence_length = 512, learning_rate = 0.001):
     emb_list = get_combined_cat_emb(gnn_sent_embeddings, sample_node_sent_maps, sequence_length)
     T5_embed_projector = nn.Linear(emb_list[0].shape[1], t5_model.config.d_model).to(device)
     
     freeze_model(t5_model)
     ## fine-tune the last layer of decoder
     