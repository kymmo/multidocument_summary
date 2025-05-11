import torch
import os
import shutil
import torch.nn as nn
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as geo_DataLoader
from torch.utils.data import DataLoader as data_DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from models.RelHetGraph import RelHetGraph
from models.DatasetLoader import EvalDataset, OptimizedDataset, custom_collate_fn
from models.CheckPointManager import ModelCheckpointManager
from models.EarlyStopper import EarlyStopper
from models.CheckPointManager import DataCheckpointManager
from models.ModelFileManager import model_fm
from utils.model_utils import freeze_model, clean_memory, print_gpu_memory

base_model = "google-t5/t5-base"
small_model = "google-t5/t5-small" #for test
t5_tokenizer = T5Tokenizer.from_pretrained(base_model)
t5_model = T5ForConditionalGeneration.from_pretrained(base_model)

def train_gnn(file_path, hidden_size, out_size, num_heads, val_file_path, t5_model = t5_model, sentence_in_size = 768, word_in_size = 768, 
               learning_rate=0.001, num_epochs=20, feat_drop=0.2, attn_drop=0.2, batch_size=32, save_method='entire_model', patience=5, sent_similarity_threshold = 0.6,
               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
     """Trains the HetGNN model using a proxy task."""
     clean_memory()
     print(f"Task runing on {device}")
     data_cpt = DataCheckpointManager()

     train_dataset = OptimizedDataset(file_path=file_path, dataset_type=data_cpt.DataType.TRAIN.value, sent_similarity=sent_similarity_threshold)
     train_dataloader = geo_DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=True,
          pin_memory=True,
          num_workers=0,
     )
     
     val_dataset = OptimizedDataset(file_path=val_file_path, dataset_type=data_cpt.DataType.VALIDATION.value, sent_similarity=sent_similarity_threshold)
     val_dataloader = geo_DataLoader(
          val_dataset,
          batch_size=batch_size,
          shuffle=False, # No shuffle for validation
          pin_memory=True,
          num_workers=0,
     )
     
     projector_hidden_size = 1024
     gnn_model = RelHetGraph(hidden_size, out_size, num_heads, sentence_in_size, word_in_size , feat_drop, attn_drop).to(device)
     t5_model = t5_model.to(device)
     T5_embed_layer_projector = nn.Sequential(
               nn.Linear(out_size, projector_hidden_size),
               nn.LayerNorm(projector_hidden_size),
               nn.ReLU(),
               nn.Linear(projector_hidden_size, t5_model.config.d_model)
     ).to(device) ## non_linear transfer
     optimizer = torch.optim.Adam(list(gnn_model.parameters()) + list(T5_embed_layer_projector.parameters()), lr=learning_rate)
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
     scaler = torch.cuda.amp.GradScaler()

     print_gpu_memory("after gnn model loading")
     
     # check point
     ckpt_mgr = ModelCheckpointManager(stage_name="gnn_model")
     early_stopper = EarlyStopper(patience=patience, checkpoint_manager=ckpt_mgr)

     start_epoch = 0
     if (checkpoint := ckpt_mgr.load(device)) is not None:
          gnn_model.load_state_dict(checkpoint['gnn_model_state'])
          T5_embed_layer_projector.load_state_dict(checkpoint['T5_projector_state'])
          optimizer.load_state_dict(checkpoint['optimizer_state'])
          if 'scheduler_state' in checkpoint:
               scheduler.load_state_dict(checkpoint['scheduler_state'])
          if 'scaler' in checkpoint:
               scaler.load_state_dict(checkpoint['scaler'])
          
          start_epoch = checkpoint['epoch'] + 1
          if 'early_stopper_state' in checkpoint and checkpoint['early_stopper_state']:
               early_stopper.load_state(checkpoint['early_stopper_state'])
               print("Found EarlyStopper state in checkpoint, loaded.")
          else:
               # if resuming from an older checkpoint saved before this feature was added
               print("No EarlyStopper state found in checkpoint. Initializing fresh.")
          print(f"Resume training! Start from epoch {start_epoch}.")

     freeze_model(t5_model)
     t5_model.eval() ## no update for T5
     last_epoch_completed = -1
     print(f"--- Training ---")
     try:
          for epoch in range(start_epoch, num_epochs):
               last_epoch_completed = epoch
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
               
               avg_train_loss = total_loss / len(train_dataloader)
               print(f"[Training] Epoch {epoch + 1} / {num_epochs}, Loss: {avg_train_loss:.4f}, Training Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
               
               # --- Validation for Early Stop ---
               print('--- Validation ---')
               gnn_model.eval()
               T5_embed_layer_projector.eval()
               total_val_loss = 0
               with torch.no_grad():
                    for batch in val_dataloader:
                         batch = batch.to(device)

                         sentence_feat = batch['sentence'].x
                         word_feat = batch['word'].x

                         with torch.cuda.amp.autocast():  # No dropout
                              sentence_embeddings = gnn_model(batch, sentence_feat, word_feat)
                              projected_embeddings = T5_embed_layer_projector(sentence_embeddings)
                              reshape_embeddings = projected_embeddings.unsqueeze(1)

                              t5_embedding_matrix = t5_model.get_input_embeddings().weight
                              similarities = chunked_cosine_similarity(projected_embeddings, t5_embedding_matrix, chunk_size=8)
                              closest_token_ids = similarities.argmax(dim=1)
                              seq_length = reshape_embeddings.size(1)
                              labels = closest_token_ids.unsqueeze(1).expand(-1, seq_length)
                              labels = labels.long().to(device)

                         outputs = t5_model(inputs_embeds=reshape_embeddings, labels=labels)
                         loss = outputs.loss
                         total_val_loss += loss.item()

               avg_val_loss = total_val_loss / len(val_dataloader)
               print(f"[Validation] Epoch {epoch + 1}/{num_epochs}, Loss: {avg_val_loss:.4f}")
     
               models_to_save = {'gnn_model': gnn_model, 'T5_projector': T5_embed_layer_projector}
               optimizers_to_save = {'optimizer': optimizer}
               schedulers_to_save = {'scheduler': scheduler}
               
               ## check point station
               current_early_stopper_state = early_stopper.get_state()
               ckpt_path = ckpt_mgr.save(
                    epoch=epoch,
                    models=models_to_save,
                    optimizers=optimizers_to_save,
                    schedulers=schedulers_to_save,
                    scaler=scaler,
                    early_stopper_state=current_early_stopper_state
               )
               print(f"[checkpoint saved] {epoch}-th epoch checkpoint has saved to path {ckpt_path}")
               
               if early_stopper(avg_val_loss, epoch, models_to_save, optimizers_to_save, schedulers_to_save, scaler):
                    break # Stop training
               
               scheduler.step()
     
     except Exception as e:
          current_epoch = last_epoch_completed if last_epoch_completed >= 0 else start_epoch -1
          emergency_path = ckpt_mgr._get_filepath(epoch=current_epoch, emergency=True)
          torch.save({
               'gnn_model_state': gnn_model.state_dict(),
               'T5_projector_state': T5_embed_layer_projector.state_dict(),
               'optimizer_state': optimizer.state_dict(),
               'scheduler_state': scheduler.state_dict(),
               'scaler': scaler.state_dict(),
               'epoch': current_epoch,
               'early_stopper_state': early_stopper.get_state(),
               'exception': str(e)
          }, emergency_path)
          print(f"[Exception] Error!! Checkpoint has saved in {emergency_path}")
          raise e
     
     best_checkpoint = ckpt_mgr.load_best(device=device)
     best_gnn_model = None
     if best_checkpoint:
          best_gnn_model = RelHetGraph(hidden_size, out_size, num_heads, sentence_in_size, word_in_size , feat_drop, attn_drop).to(device)
          best_gnn_model.load_state_dict(best_checkpoint['gnn_model_state'])
          T5_embed_layer_projector.load_state_dict(best_checkpoint['T5_projector_state'])
          gnn_model = best_gnn_model
          print(f"[Checkpoint] The least-loss GNN model (from epoch {best_checkpoint.get('epoch', 'N/A')}) is reloaded from checkpoint.")
     
     if save_method == 'entire_model':
          ## save entire model
          model_fm.save_gnn(gnn_model)
          # torch.save(gnn_model, 'gnn_trained_weights.pt')
     elif save_method == 'weights':
          torch.save(gnn_model.state_dict(), 'gnn_trained_weights.pth')
          torch.save(T5_embed_layer_projector.state_dict(), 't5_projector_weights.pth')
     else:
          raise ValueError(f'save_method has no value as {save_method}.')
     
     print("--- GNN Training Finish! ---")
     del gnn_model, t5_model, best_gnn_model
     del T5_embed_layer_projector
     del optimizer, scheduler, scaler
     clean_memory()

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