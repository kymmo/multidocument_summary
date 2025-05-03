import torch
import os
from pathlib import Path
import torch.nn as nn
import time
import math
import shutil
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader as data_DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, get_linear_schedule_with_warmup 

from models.DatasetLoader import EvalDataset, custom_collate_fn
from models.CustomT5 import CustomT5
from models.gnn_train_t5 import train_gnn
from models.CheckPointManager import ModelCheckpointManager, DataCheckpointManager
from utils.model_utils import freeze_model, clean_memory, print_gpu_memory
from models.LongTextEncoder import LongTextEncoder
from models.EarlyStopper import EarlyStopper
from models.ModelFileManager import model_fm

base_model = "google-t5/t5-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_tokenizer = T5Tokenizer.from_pretrained(base_model)
t5_model = T5ForConditionalGeneration.from_pretrained(base_model, use_cache=False).to(device)
t5_model.gradient_checkpointing_enable()

def train_gnn_t5(dataset_path, hidden_size, out_size, num_heads=8, learning_rate=0.001, num_epochs=20, 
               feat_drop=0.1, attn_drop=0.1, batch_size=16, patience=5, sent_similarity_threshold=0.6, 
               learning_rates_dict = None, warmup_ratio=0.1):
     ## gnn training, t5 freezed
     train_data_path = os.path.join(dataset_path, "train.jsonl")
     val_data_path = os.path.join(dataset_path, "validation.jsonl")
     ## path check
     if not Path(train_data_path).exists() or not Path(val_data_path).exists():
          raise FileNotFoundError(f"File path {train_data_path} or {val_data_path} is not exist!")
     print(f"Accessing training data path: {train_data_path} and validation data path: {val_data_path}")
     
     print(f"Start training GNN. Parameters: hidden_size: {hidden_size}, out_size: {out_size}, attention_heads: {num_heads}")
     gnn_start_time = time.time()
     #### train gnn, freeze t5
     train_gnn(
          file_path=train_data_path,
          val_file_path=val_data_path,
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
          save_method='entire_model',
          patience=patience,
          sent_similarity_threshold=sent_similarity_threshold,
     )
     gnn_end_time = time.time()
     print(f"Finish gnn training, time cost:  {gnn_end_time - gnn_start_time:.4f} s.")
     
     #### train t5, freeze gnn
     print("Start fine-tuning T5...")
     t5_start_time = time.time()
     fine_tune_t5(
          file_path=train_data_path,
          val_file_path=val_data_path,
          out_size=out_size,
          num_epochs=num_epochs,
          batch_size=batch_size,
          patience=patience,
          sent_similarity_threshold=sent_similarity_threshold,
          learning_rates_dict=learning_rates_dict,
          warmup_ratio=warmup_ratio,
     )
     t5_end_time = time.time()
     print(f"Finish T5 fine-tune, time cost:  {t5_end_time - t5_start_time:.4f} s.")
     
     print("*** Two-stage training finish! ***")

def get_combined_embed2(batch_graph_list, gnn_embeddings, sent_text):
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

def fine_tune_t5(file_path, val_file_path, out_size, num_epochs = 20, 
               batch_size = 8, accumulate_step = 4, patience = 5, sent_similarity_threshold=0.6,
               learning_rates_dict=None, warmup_ratio=0.1):
     ## data load
     data_cpt = DataCheckpointManager()
     train_dataset = EvalDataset(file_path=file_path, dataset_type=data_cpt.DataType.TRAIN.value, sent_similarity=sent_similarity_threshold)
     train_dataloader = data_DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=True,
          # pin_memory=True, ## data has been in GPU while training gnn
          collate_fn=custom_collate_fn
     )

     val_dataset = EvalDataset(file_path=val_file_path, dataset_type=data_cpt.DataType.VALIDATION.value, sent_similarity=sent_similarity_threshold)
     val_dataloader = data_DataLoader(
          val_dataset, batch_size=batch_size, shuffle=False, # No shuffle
          collate_fn=custom_collate_fn
     )
     
     ckpt_mgr = ModelCheckpointManager(stage_name="custom_t5")
     early_stopper = EarlyStopper(patience=patience, checkpoint_manager=ckpt_mgr)
     
     ## models load
     try:
          # gnn_model = torch.load('gnn_trained_weights.pt')
          gnn_model = model_fm.load_gnn()
          gnn_model = gnn_model.to(device)
          gnn_model.eval()
          freeze_model(gnn_model)
     except FileNotFoundError:
          raise FileNotFoundError("gnn_trained_weights.pt (or .pth) not found. Run GNN training first.")
     except Exception as e:
          raise RuntimeError(f"Error loading GNN model: {e}")
     
     config = T5Config.from_pretrained(base_model)
     config.projector_input_size = out_size + t5_model.config.hidden_size # concated emb size
     custom_t5_model = CustomT5(config).to(device)
     if learning_rates_dict is None or learning_rates_dict["encoder_last2"] is None or learning_rates_dict["decoder_last2"] is None \
          or learning_rates_dict["projector"] is None:
          learning_rates_dict = {
               "encoder_last2": 1e-4,
               "decoder_last2": 1e-4,
               "projector": 1e-3
          }
          
     optimizer = torch.optim.AdamW(
          [
          {"params": custom_t5_model.encoder.block[-2:].parameters(), "lr": learning_rates_dict["encoder_last2"]},
          {"params": custom_t5_model.decoder.block[-2:].parameters(), "lr": learning_rates_dict["decoder_last2"]},
          {"params": custom_t5_model.projector.parameters(), "lr": learning_rates_dict["projector"]}
          ],
          weight_decay=0.01
     )
     # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
     
     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accumulate_step)
     max_train_steps = num_epochs * num_update_steps_per_epoch
     num_warmup_steps = int(max_train_steps * warmup_ratio)
     print(f"[Scheduler] Total training steps estimated: {max_train_steps}, Warmup steps: {num_warmup_steps}")
     scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=num_warmup_steps,
          num_training_steps=max_train_steps
     )
     scaler = torch.cuda.amp.GradScaler(enabled=True)

     print_gpu_memory("after t5 model loading")
     resume = False
     start_epoch = 0
     accumulated_batches = 0
     global_step = 0
     if (checkpoint := ckpt_mgr.load(device)) is not None:
          custom_t5_model.load_state_dict(checkpoint['custom_t5_model_state'])
          optimizer.load_state_dict(checkpoint['optimizer_state'])
          if 'scheduler_state' in checkpoint:
               scheduler.load_state_dict(checkpoint['scheduler_state'])
          if 'scaler' in checkpoint:
               scaler.load_state_dict(checkpoint['scaler'])
          
          start_epoch = checkpoint['epoch']
          accumulated_batches = checkpoint.get('accumulated_batches', 0)
          global_step = checkpoint.get('global_step', 0)
          resume = True
          if accumulated_batches >= len(train_dataloader): ## new epoch
               accumulated_batches = 0
               start_epoch += 1

          if 'early_stopper_state' in checkpoint and checkpoint['early_stopper_state']:
               early_stopper.load_state(checkpoint['early_stopper_state'])
               print("Found EarlyStopper state in checkpoint, loaded.")
          else:
               # if resuming from an older checkpoint saved before this feature was added
               print("No EarlyStopper state found in checkpoint. Initializing fresh.")
          
          print(f"Resume training! From epoch {start_epoch}, batch {accumulated_batches}.")
     
     try:
          for epoch in range(start_epoch, num_epochs):
               print(f"--- Training ---")
               custom_t5_model.train()
               total_loss = 0.0
               processed_batches_this_epoch = 0
               skip_batches = accumulated_batches if resume and epoch == start_epoch else 0
               optimizer.zero_grad()

               for batch_idx, batch in enumerate(train_dataloader):
                    if batch_idx < skip_batches:
                         continue
               
                    batch_graph, batch_map, batch_summary = batch
                    batched_graph = Batch.from_data_list(batch_graph).to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast():
                         sentence_feat = batched_graph['sentence'].x
                         word_feat = batched_graph['word'].x
                         sent_text = batched_graph['sentence'].text
                         
                         ## adding data noise
                         corrupted_sentence_feat = F.dropout(sentence_feat, p=0.1, training=custom_t5_model.training)

                         # forward
                         gnn_embeddings = gnn_model(batched_graph, corrupted_sentence_feat, word_feat)
                         concat_embs_list = get_combined_embed2(batch_graph, gnn_embeddings, sent_text)
                         
                         outputs = custom_t5_model(combin_embeddings_list = concat_embs_list, label_summaries=batch_summary)
                         loss = outputs.loss
                         scaled_loss  = loss / accumulate_step
                         
                    scaler.scale(scaled_loss).backward()
                    total_loss += loss.item()
                    processed_batches_this_epoch += 1
                    
                    ## gradient accumulate
                    if ((batch_idx + 1) % accumulate_step == 0) or (batch_idx + 1 >= len(train_dataloader)):
                         scaler.step(optimizer)
                         scaler.update()
                         global_step += 1
                         scheduler.step()
                         optimizer.zero_grad()
                         
                         if global_step % 100 == 0: # Log every 100 steps example
                              print(f"[T5 Scheduler] Optimize Step {global_step}, Current LR: {scheduler.get_last_lr()[0]:.8f}")
                         
               avg_loss = total_loss / processed_batches_this_epoch if processed_batches_this_epoch > 0 else 0
               print(f"[Training] Epoch {epoch+1} / {num_epochs}, Loss: {avg_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
               
               # --- Validation for Early Stop ---
               print('--- Validation ---')
               custom_t5_model.eval()
               total_val_loss = 0.0
               num_val_batches = 0
               with torch.no_grad():
                    for val_batch in val_dataloader:
                         val_graph, val_map, val_summary = val_batch
                         
                         batched_graph = Batch.from_data_list(val_graph).to(device, non_blocking=True)
                         with torch.cuda.amp.autocast():
                              sentence_feat = batched_graph['sentence'].x
                              word_feat = batched_graph['word'].x
                              sent_text = batched_graph['sentence'].text
                              gnn_embeddings = gnn_model(batched_graph, sentence_feat, word_feat)
                              concat_embs_list = get_combined_embed2(val_graph, gnn_embeddings, sent_text)
                              
                              outputs = custom_t5_model(combin_embeddings_list=concat_embs_list, label_summaries=val_summary)
                              total_val_loss += outputs.loss.item()
                              num_val_batches += 1
               
               avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
               print(f"[Validation] Epoch {epoch+1}/{num_epochs}, Loss: {avg_val_loss:.4f}")

               # --- Early Stopping Check ---
               models_to_save = {'custom_t5_model': custom_t5_model}
               optimizers_to_save = {'optimizer': optimizer}
               schedulers_to_save = {'scheduler': scheduler}
               if early_stopper(avg_val_loss, epoch, models_to_save, optimizers_to_save, schedulers_to_save, scaler, global_step):
                    break # Stop training
               
               # scheduler.step()
               
               ckpt_path = ckpt_mgr.save(
                    epoch=epoch,
                    models=models_to_save,
                    optimizers=optimizers_to_save,
                    schedulers=schedulers_to_save,
                    scaler=scaler,
                    accumulated_batches= len(train_dataloader),
                    global_step=global_step,
                    early_stopper_state=early_stopper.get_state(),
               )
               print(f"[checkpoint saved] {epoch}-th epoch checkpoint has saved to path {ckpt_path}")
                              
               if resume and epoch == start_epoch:
                    resume = False
                    accumulated_batches = 0
     
     except Exception as e:
          emergency_path = ckpt_mgr._get_filepath(emergency=True)
          current_batch_index = skip_batches + processed_batches_this_epoch
          torch.save({
               'custom_t5_model_state': custom_t5_model.state_dict(),
               'optimizer_state': optimizer.state_dict(),
               'scheduler_state': scheduler.state_dict(),
               'scaler': scaler.state_dict(),
               'epoch': epoch,
               'accumulated_batches': current_batch_index,
               'global_step': global_step,
               'early_stopper_state': early_stopper.get_state(),
               'exception': str(e)
          }, emergency_path)
          print(f"[Exception] Error!! Checkpoint has saved in {emergency_path}")
          raise e
     
     best_checkpoint = ckpt_mgr.load_best(device=device)
     if best_checkpoint:
          custom_t5_model.load_state_dict(best_checkpoint['custom_t5_model_state'])
          best_global_step = best_checkpoint.get('global_step', 'N/A')
          print(f"[Checkpoint] Best T5 model state (from epoch {best_checkpoint.get('epoch', 'N/A')}, step {best_global_step}) loaded.")
     else:
          print("[Checkpoint] Best checkpoint not found. Using the model state from the last completed epoch.")
     
     # custom_t5_model.save_pretrained("./fine_tuned_t5")
     model_fm.save_t5(custom_t5_model)
     
     del custom_t5_model
     del gnn_model
     clean_memory()
