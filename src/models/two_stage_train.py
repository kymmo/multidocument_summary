import torch
import os
from pathlib import Path
import torch.nn as nn
import time
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader as data_DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

from models.DatasetLoader import EvalDataset, OptimizedDataset, custom_collate_fn
from models.CustomT5 import CustomT5
from models.gnn_train_t5 import train_gnn
from models.CheckPointManager import ModelCheckpointManager
from utils.model_utils import freeze_model, clean_memory, print_gpu_memory
from models.LongTextEncoder import LongTextEncoder

base_model = "google-t5/t5-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_tokenizer = T5Tokenizer.from_pretrained(base_model)
t5_model = T5ForConditionalGeneration.from_pretrained(base_model, use_cache=False).to(device)
t5_model.gradient_checkpointing_enable()

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

def get_combined_embed2(batch_graph_list, gnn_embeddings, sent_text):
     """改进版特征融合函数（显存优化+长文本支持）"""
     concat_embedding_list = []
     start_ind = 0
     encoder = LongTextEncoder(t5_tokenizer, t5_model)
     
     ## cal doc_emb first
     process_doc = []
     for graph_sents in sent_text:
          prompt = "Generate a summary from documents' embeddings: "
          full_doc = prompt + " ".join(graph_sents)
          process_doc.append(full_doc)
          
     with torch.no_grad(), torch.cuda.amp.autocast():
          docs_embs = encoder.encode_batch(process_doc, batch_size=16)
     
     ## cal sent level t5 emb
     graph_sent_embs = []
     for graph_sents in sent_text:
          process_sents = []
          for sent_id, sent in enumerate(graph_sents):
               # previous 3 sents as context, adapt most 500 chars.
               cont_start = max(sent_id - 3, 0)
               context = " ".join(graph_sents[cont_start:sent_id])[:500]
               cont_sent = f"[Context: {context}] {sent}"
               process_sents.append(cont_sent)
               
          with torch.no_grad(), torch.cuda.amp.autocast():
               sent_embs = encoder.encode_batch(process_sents, batch_size=16)
          
          graph_sent_embs.append(sent_embs)

     for i_th, graph in enumerate(batch_graph_list): # for each embs of graph
          graph_sent_num = graph['sentence'].x.shape[0]
          gnn_sent_embs = gnn_embeddings[start_ind: start_ind + graph_sent_num]
          start_ind += graph_sent_num
          
          with torch.no_grad():
               gnn_norm = F.normalize(gnn_sent_embs, p=2, dim=-1)
               t5_norm = F.normalize(graph_sent_embs[i_th].to(device), p=2, dim=-1)
               doc_emb = docs_embs[i_th].unsqueeze(0).to(device)
               
               # fuse the whole doc infor
               fused_gnn = gnn_norm + 0.1 * doc_emb
               fused_t5 = t5_norm + 0.1 * doc_emb
               
               combined = torch.cat([fused_gnn, fused_t5], dim=-1)
               concat_embedding_list.append(combined)
          
          del gnn_norm, t5_norm, fused_gnn, doc_emb, fused_t5
          clean_memory()
     
     return concat_embedding_list
     
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
                    ## get T5 embeddings
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
                    
                    del input_ids, attention_mask

                    ## concatinate GNN and T5 embedding
                    gnn_emb_norm = nn.LayerNorm(gnn_sent_embs.shape[1], device=device)(padding_gnn_embeddings)
                    t5_emb_norm = nn.LayerNorm(t5_model.config.hidden_size, device=device)(avg_t5_embeddings)
                    
                    combined_embeddings = torch.cat([gnn_emb_norm, t5_emb_norm], dim=1)
                    
                    concat_embedding_list.append(combined_embeddings)
                    
                    del combined_embeddings
     
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

def fine_tune_t5(file_path, out_size, num_epochs = 20, batch_size = 8, accumulate_step = 4):
     ## data load
     train_dataset = EvalDataset(file_path)
     train_dataloader = data_DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=True,
          # pin_memory=True, ## data has been in GPU while training gnn
          collate_fn=custom_collate_fn
     )
     
     ckpt_mgr = ModelCheckpointManager(stage_name="custom_t5")
     
     ## models load
     gnn_model = torch.load('gnn_trained_weights.pt')
     gnn_model.eval()
     freeze_model(gnn_model)
     
     config = T5Config.from_pretrained(base_model)
     config.projector_input_size = out_size + t5_model.config.hidden_size # concated emb size
     custom_t5_model = CustomT5(config).to(device)
     custom_t5_model.train()
     optimizer = torch.optim.AdamW(
          [
          {"params": custom_t5_model.encoder.block[-2:].parameters(), "lr": 1e-4},
          {"params": custom_t5_model.decoder.block[-2:].parameters(), "lr": 1e-4},
          {"params": custom_t5_model.projector.parameters(), "lr": 1e-3}
          ],
          weight_decay=0.01
     )
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
     scaler = torch.cuda.amp.GradScaler(enabled=True)

     print_gpu_memory("after t5 model loading")
     
     torch.cuda.empty_cache()
     resume = False
     if (checkpoint := ckpt_mgr.load(device)) is not None:
          custom_t5_model.load_state_dict(checkpoint['custom_t5_model_state'])
          optimizer.load_state_dict(checkpoint['optimizer_state'])
          scheduler.load_state_dict(checkpoint['scheduler_state'])
          scaler.load_state_dict(checkpoint['scaler'])
          
          start_epoch = checkpoint['epoch']
          accumulated_batches = checkpoint.get('accumulated_batches', 0)
          resume = True
          if accumulated_batches >= len(train_dataloader): ## new epoch
               accumulated_batches = 0
               start_epoch += 1
               
          print(f"Resume training! From epoch {start_epoch}, batch {accumulated_batches}.")
          
     print(f"Setting finish. Start training epoch...")
     try:
          for epoch in range(start_epoch if resume else 0, num_epochs):
               total_loss = 0.0
               actual_batch_count = 0
               
               optimizer.zero_grad()
               skip_batches = accumulated_batches if resume else 0

               for batch_idx, batch in enumerate(train_dataloader):
                    if batch_idx < skip_batches:
                         continue
               
                    batch_graph, batch_map, batch_summary = batch
                    
                    with torch.cuda.amp.autocast():
                         batched_graph = Batch.from_data_list(batch_graph).to(device, non_blocking=True)
                         sentence_feat = batched_graph['sentence'].x
                         word_feat = batched_graph['word'].x
                         sent_text = batched_graph['sentence'].text
                         
                         ## adding data noise
                         corrupted_sentence_feat = F.dropout(sentence_feat, p=0.1, training=gnn_model.training)

                         # forward
                         gnn_embeddings = gnn_model(batched_graph, corrupted_sentence_feat, word_feat)
                         concat_embs_list = get_combined_embed2(batch_graph, gnn_embeddings, sent_text)
                         
                         outputs = custom_t5_model(combin_embeddings_list = concat_embs_list, label_summaries=batch_summary)
                         loss = outputs.loss
                         loss = loss / accumulate_step
                         
                    scaler.scale(loss).backward()
                    total_loss += loss.item() * accumulate_step
                    ## gradient accumulate
                    if ((batch_idx + 1) % accumulate_step == 0) or (batch_idx + 1 >= len(train_dataloader)):
                         actual_batch_count += 1
                         scaler.step(optimizer)
                         scaler.update()
                         optimizer.zero_grad()
                         
                         ## check point
                         ckpt_path = ckpt_mgr.save(
                              epoch=epoch,
                              models={'custom_t5_model': custom_t5_model},
                              optimizers={'optimizer': optimizer},
                              schedulers={'scheduler': scheduler},
                              scaler=scaler,
                              accumulated_batches= batch_idx + 1
                         )
                         print(f"[checkpoint saved] {epoch}-th epoch, {batch_idx + 1}-th batch checkpoint has saved to path {ckpt_path}")
               
               avg_loss = total_loss / (actual_batch_count * accumulate_step) if actual_batch_count > 0 else 0
               print(f"Epoch {epoch+1} / {num_epochs}, Loss: {avg_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
               scheduler.step()
               
               if resume: ## reset new state
                    resume = False
                    accumulated_batches = 0
     
     except Exception as e:
          emergency_path = ckpt_mgr._get_filepath(emergency=True)
          torch.save({
               'custom_t5_model_state': custom_t5_model.state_dict(),
               'optimizer_state': optimizer.state_dict(),
               'scheduler_state': scheduler.state_dict(),
               'scaler': scaler.state_dict(),
               'epoch': epoch,
               'accumulated_batches': batch_idx,
               'exception': str(e)
          }, emergency_path)
          print(f"[Exception] Error!! Checkpoint has saved in {emergency_path}")
          raise e
          
     custom_t5_model.save_pretrained("./fine_tuned_t5")
     
     del custom_t5_model
     del gnn_model
     clean_memory()
