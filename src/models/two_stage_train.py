import torch
import os
from pathlib import Path
import time
import math
import threading
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Batch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader as data_DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5TokenizerFast, get_cosine_schedule_with_warmup

from models.DatasetLoader import EvalDataset, custom_collate_fn
from models.CustomT5 import CustomT5
from models.gnn_train_t5 import train_gnn
from models.CheckPointManager import ModelCheckpointManager, DataType
from utils.model_utils import freeze_model, clean_memory, print_gpu_memory, print_and_save_loss_curve, monitor_usage
from models.CustomEncoder import LongTextEncoder
from models.EarlyStopper import EarlyStopper
from models.ModelFileManager import model_fm

base_model = "google-t5/t5-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_tokenizer = T5TokenizerFast.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained(base_model, use_cache=False).to(device)
t5_model.gradient_checkpointing_enable()

def train_gnn_t5(dataset_path, hidden_size, out_size, num_heads=8, learning_rate=0.001, num_epochs=20, 
               feat_drop=0.1, attn_drop=0.1, gnn_batch_size=16, llm_batch_size=4, 
               patience=5, llm_accumulate_step=4, sent_similarity_threshold=0.6, gnn_accumulation_steps=4,
               learning_rates_dict = None, warmup_ratio=0.1):
     ## gnn training, t5 freezed
     train_data_path = os.path.join(dataset_path, "train.jsonl")
     val_data_path = os.path.join(dataset_path, "validation.jsonl")
     ## path check
     if not Path(train_data_path).exists() or not Path(val_data_path).exists():
          raise FileNotFoundError(f"File path {train_data_path} or {val_data_path} is not exist!")
     
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
          projection_dim = 768,
          learning_rate=learning_rate,
          num_epochs=num_epochs,
          feat_drop=feat_drop,
          attn_drop=attn_drop,
          batch_size=gnn_batch_size,
          save_method='entire_model',
          patience=patience,
          warmup_ratio=warmup_ratio,
          gnn_accumulation_steps=gnn_accumulation_steps,
          sent_similarity_threshold=sent_similarity_threshold,
     )
     gnn_end_time = time.time()
     print(f"** Finish gnn training, time cost:  {gnn_end_time - gnn_start_time:.4f} s.")
     
     #### train t5, freeze gnn
     print("Start fine-tuning T5...")
     t5_start_time = time.time()
     fine_tune_t5(
          file_path=train_data_path,
          val_file_path=val_data_path,
          out_size=out_size,
          num_epochs=num_epochs,
          batch_size=llm_batch_size,
          accumulate_step=llm_accumulate_step,
          patience=patience,
          sent_similarity_threshold=sent_similarity_threshold,
          learning_rates_dict=learning_rates_dict,
          warmup_ratio=warmup_ratio,
     )
     t5_end_time = time.time()
     print(f"** Finish T5 fine-tune, time cost:  {t5_end_time - t5_start_time:.4f} s.")
     
     print("*** Two-stage training finish! ***")

def get_combined_embed2(batch_graph_list, sentence_graph_embs, sentence_text_embs):
     """_summary_ concate sentence_graph_embs and sentence_text_embs correspondingly.

     Args:
          batch_graph_list (_type_): original hetrograph list.
          sentence_graph_embs (_type_): corresponding original graph sentence embedding, sorting by id.
          sentence_text_embs (_type_): LLM text embeddings with PROMPT + graph corresponding sentence text embedding.

     Returns:
          _type_: return the combined tensor
     """
     concat_embedding_list = []
     start_ind = 0
     global_u, global_v = None, None
     text_start_index = 1
     
     torch.manual_seed(24)
     doc_discrimination_strength = 1.5
     max_doc_id = 100
     
     for i_th, graph in enumerate(batch_graph_list):
          graph_sent_num = graph['sentence'].x.shape[0]
          gnn_sent_embs = sentence_graph_embs[start_ind: start_ind + graph_sent_num]
          start_ind += graph_sent_num
          t5_sent_embs = sentence_text_embs[text_start_index: text_start_index + graph_sent_num]
          text_start_index += graph_sent_num
          
          gnn_mean = gnn_sent_embs.mean(dim=0)
          gnn_std = gnn_sent_embs.std(dim=0) + 1e-8
          gnn_sent_embs = (gnn_sent_embs - gnn_mean) / gnn_std
          
          gnn_prompt_emb = torch.zeros_like(gnn_sent_embs[0].unsqueeze(0))  # (1, dim)
          text_prompt_emb = sentence_text_embs[0].unsqueeze(0)  # (1, dim)
          
          if global_u is None:
               global_u = torch.randn(gnn_sent_embs.shape[-1], device=gnn_sent_embs.device)
               global_u = global_u / torch.norm(global_u)
               
               global_v = torch.randn(gnn_sent_embs.shape[-1], device=gnn_sent_embs.device)
               global_v = global_v - torch.dot(global_v, global_u) * global_u
               global_v = global_v / torch.norm(global_v)
          
          doc_id = i_th % max_doc_id
          angle = doc_id * 2 * torch.pi / (max_doc_id / doc_discrimination_strength)
          
          u_component = torch.einsum('bi,i->b', gnn_sent_embs, global_u).unsqueeze(1) * global_u.unsqueeze(0)
          v_component = torch.einsum('bi,i->b', gnn_sent_embs, global_v).unsqueeze(1) * global_v.unsqueeze(0)
          other_component = gnn_sent_embs - u_component - v_component
          
          rotation_strength = 0.4 + 0.8 * (doc_id % 5) / 4  # [0.4, 1.2]
          
          cos_angle = torch.cos(torch.tensor(angle * rotation_strength))
          sin_angle = torch.sin(torch.tensor(angle * rotation_strength))
          
          gnn_rotated = (
               cos_angle * u_component +
               sin_angle * v_component +
               other_component
          )
          
          gnn_combined = torch.cat([
               gnn_prompt_emb,
               gnn_rotated
          ], dim=0)
          
          t5_mean = t5_sent_embs.mean(dim=0)
          t5_std = t5_sent_embs.std(dim=0) + 1e-8
          t5_sent_embs = (t5_sent_embs - t5_mean) / t5_std
          
          text_combined = torch.cat([
               text_prompt_emb,
               t5_sent_embs
          ], dim=0)
          
          part1 = torch.ones(10, device=gnn_sent_embs.device) * (doc_id % 10) / 80.0
          part2 = torch.eye(10, device=gnn_sent_embs.device)[doc_id % 10].float() * 0.08
          part3 = torch.eye(10, device=gnn_sent_embs.device)[doc_id // 10].float() * 0.08
          doc_bias = torch.cat([part1, part2, part3])
          
          if doc_bias.dim() == 1:
               doc_bias = doc_bias.unsqueeze(0)
          
          doc_bias = doc_bias.repeat(gnn_combined.shape[0], 1)
          
          combined = torch.cat([
               gnn_combined,
               text_combined,
               doc_bias
          ], dim=-1)
          
          layer_norm = torch.nn.LayerNorm(combined.shape[-1], elementwise_affine=False)
          combined = layer_norm(combined)
          
          concat_embedding_list.append(combined)
          
     return concat_embedding_list

def fine_tune_t5(file_path, val_file_path, out_size, num_epochs = 20, 
               batch_size = 8, accumulate_step = 4, patience = 5, sent_similarity_threshold=0.6,
               learning_rates_dict=None, warmup_ratio=0.1):
     ## data load
     train_dataset = EvalDataset(file_path=file_path, dataset_type=DataType.TRAIN.value, sent_similarity=sent_similarity_threshold)
     train_dataloader = data_DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=True,
          num_workers=0,
          pin_memory=True,
          collate_fn=custom_collate_fn
     )

     val_dataset = EvalDataset(file_path=val_file_path, dataset_type=DataType.VALIDATION.value, sent_similarity=sent_similarity_threshold)
     val_dataloader = data_DataLoader(
          val_dataset, batch_size=batch_size, shuffle=False, # No shuffle
          num_workers=0,
          pin_memory=True,
          collate_fn=custom_collate_fn
     )
     
     ckpt_mgr = ModelCheckpointManager(stage_name="custom_t5")
     early_stopper = EarlyStopper(patience=patience, checkpoint_manager=ckpt_mgr)
     
     try:
          gnn_model = model_fm.load_gnn()
          gnn_model = gnn_model.to(device)
          gnn_model.eval()
          freeze_model(gnn_model)
     except FileNotFoundError:
          raise FileNotFoundError("gnn_trained_weights.pt (or .pth) not found. Run GNN training first.")
     except Exception as e:
          raise RuntimeError(f"Error loading GNN model: {e}")
     
     config = T5Config.from_pretrained(base_model)
     config.dropout_rate = 0.05
     config.projector_input_size = out_size + t5_model.config.hidden_size + 30 # concated emb size + doc_emb_size
     config.torch_dtype = torch.float16
     config.gradient_checkpointing = True
     custom_t5_model = CustomT5(config).to(device)
     if learning_rates_dict is None or learning_rates_dict["shallow_layers"] is None or learning_rates_dict["deep_layers"] is None \
          or learning_rates_dict["projector"] is None:
          learning_rates_dict = {
               "shallow_layers": 1e-4,
               "deep_layers": 5e-5,
               "projector": 1e-3,
          }
          
     optimizer_grouped_parameters = [
          {
               "params": custom_t5_model.projector.parameters(),
               "lr": learning_rates_dict["projector"]
          },
          {
               "params": custom_t5_model.encoder.block[-2:].parameters(),
               "lr": learning_rates_dict["shallow_layers"]
          },
          {
               "params": custom_t5_model.encoder.block[-4:-2].parameters(),
               "lr": learning_rates_dict["deep_layers"]
          },
          {
               "params": custom_t5_model.decoder.block[-2:].parameters(),
               "lr": learning_rates_dict["shallow_layers"]
          },
          {
               "params": custom_t5_model.decoder.block[-4:-2].parameters(),
               "lr": learning_rates_dict["deep_layers"]
          }
     ]
     optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
     
     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accumulate_step)
     max_train_steps = num_epochs * num_update_steps_per_epoch
     num_warmup_steps = int(max_train_steps * warmup_ratio)
     print(f"[Scheduler] Total training steps estimated: {max_train_steps}, Warmup steps: {num_warmup_steps}")
     scheduler = get_cosine_schedule_with_warmup(
          optimizer,
          num_warmup_steps=num_warmup_steps,
          num_training_steps=max_train_steps,
          num_cycles=1.0,
          last_epoch=-1,
     )
     scaler = torch.cuda.amp.GradScaler(enabled=True)

     print_gpu_memory("after t5 model loading")
     resume = False
     start_epoch = 0
     global_step = 0
     train_losses = [] # list to store training losses for plotting
     val_losses = []
     if (checkpoint := ckpt_mgr.load(device)) is not None:
          custom_t5_model.load_state_dict(checkpoint['custom_t5_model_state'])
          optimizer.load_state_dict(checkpoint['optimizer_state'])
          if 'scheduler_state' in checkpoint:
               scheduler.load_state_dict(checkpoint['scheduler_state'])
          if 'scaler' in checkpoint:
               scaler.load_state_dict(checkpoint['scaler'])
          
          start_epoch = checkpoint['epoch'] + 1
          global_step = checkpoint.get('global_step', 0)
          resume = True

          if 'early_stopper_state' in checkpoint and checkpoint['early_stopper_state']:
               early_stopper.load_state(checkpoint['early_stopper_state'])
               print("Found EarlyStopper state in checkpoint, loaded.")
          else:
               print("No EarlyStopper state found in checkpoint. Initializing fresh.")
          
          if 'train_losses' in checkpoint: train_losses = checkpoint['train_losses']
          if 'val_losses' in checkpoint: val_losses = checkpoint['val_losses']
          
          print(f"Resume training! From epoch {start_epoch}.")
     
     if early_stopper.early_stop_triggered:
          print("Early stop has been triggered. Skip training.")
          start_epoch = num_epochs
          
     try:
          long_text_encoder = LongTextEncoder(t5_tokenizer, t5_model)
          
          stop_event = threading.Event()
          monitor_thread = threading.Thread(target=monitor_usage, args=(1, stop_event, "T5 Training"))
          monitor_thread.start()
     
          for epoch in range(start_epoch, num_epochs):
               print(f"--- T5 Fine-tune ---")
               custom_t5_model.train()
               total_loss = 0.0
               processed_batches_this_epoch = 0
               optimizer.zero_grad()

               for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"{epoch}-th Train Epoch"):
                    batch_graph, batch_map, batch_summary = batch
                    batched_graph = Batch.from_data_list(batch_graph).to(device, non_blocking=True)
                    
                    with torch.no_grad():
                         sentence_graph_embs, _ = gnn_model(batched_graph)
                         sentence_graph_embs = sentence_graph_embs.detach()
                    
                    sent_texts = batched_graph['sentence'].text
                    sent_text_list = [sent for doc in sent_texts for sent in doc]
                    prompt = "Summarize: "
                    sent_text_list.insert(0, prompt)
                    sentence_text_embs = long_text_encoder(sent_text_list)
                    
                    concat_embs_list = get_combined_embed2(batch_graph, sentence_graph_embs, sentence_text_embs)
                    
                    with torch.cuda.amp.autocast():
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
                         optimizer.zero_grad(set_to_none=True)
                         
                         if global_step % 100 == 0:
                              current_lrs = scheduler.get_last_lr()
                              lr_strings = [f"{lr:.8f}" for lr in current_lrs]
                              print(f"[Batch Update] Batch {batch_idx}, Global Step: {global_step}, Learning Rates: {lr_strings}")
     
                    del batched_graph, sentence_graph_embs, sentence_text_embs, concat_embs_list, outputs
                    clean_memory()
                         
               avg_train_loss = total_loss / processed_batches_this_epoch if processed_batches_this_epoch > 0 else 0
               train_losses.append(avg_train_loss)
               current_lrs = scheduler.get_last_lr()
               lr_strings = [f"{lr:.8f}" for lr in current_lrs]
               print(f"[Training] Epoch {epoch+1} / {num_epochs}, Loss: {avg_train_loss:.4f}, Learning Rates: {lr_strings}")
               
               # --- Validation for Early Stop ---
               print('--- T5 Validation ---')
               custom_t5_model.eval()
               total_val_loss = 0.0
               num_val_batches = 0
               with torch.no_grad():
                    for val_batch_id, val_batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"{epoch}-th Val Epoch"):
                         val_graph, val_map, val_summary = val_batch
                         batched_graph = Batch.from_data_list(val_graph).to(device, non_blocking=True)
                         
                         sentence_graph_embs, _ = gnn_model(batched_graph)
                         sentence_graph_embs = sentence_graph_embs.detach()
                         sent_texts = batched_graph['sentence'].text
                         sent_text_list = [sent for doc in sent_texts for sent in doc]
                         prompt = "Summarize: "
                         sent_text_list.insert(0, prompt)
                         sentence_text_embs = long_text_encoder(sent_text_list)

                         concat_embs_list = get_combined_embed2(
                              val_graph,
                              sentence_graph_embs,
                              sentence_text_embs,
                         )

                         with torch.cuda.amp.autocast():
                              outputs = custom_t5_model(combin_embeddings_list=concat_embs_list, label_summaries=val_summary)
                              total_val_loss += outputs.loss.item()
                              num_val_batches += 1

                         del batched_graph, sentence_graph_embs, sentence_text_embs, concat_embs_list, outputs
                         clean_memory()

               avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
               val_losses.append(avg_val_loss)
               print(f"[Validation] Epoch {epoch+1}/{num_epochs}, Loss: {avg_val_loss:.4f}")

               # --- Early Stopping Check ---
               models_to_save = {'custom_t5_model': custom_t5_model}
               optimizers_to_save = {'optimizer': optimizer}
               schedulers_to_save = {'scheduler': scheduler}
                              
               ckpt_path = ckpt_mgr.save(
                    epoch=epoch, 
                    models=models_to_save,
                    optimizers=optimizers_to_save,
                    schedulers=schedulers_to_save,
                    scaler=scaler,
                    global_step=global_step,
                    early_stopper_state=early_stopper.get_state(),
                    train_losses=train_losses,
                    val_losses=val_losses
               )
               print(f"[checkpoint saved] {epoch}-th epoch checkpoint has saved to path {ckpt_path}")
               
               if early_stopper(avg_val_loss, epoch, models_to_save, optimizers_to_save, 
                              schedulers_to_save, scaler, global_step,
                              train_losses_history=train_losses, val_losses_history=val_losses):
                    break # Stop training
               
               if resume and epoch == start_epoch:
                    resume = False
     
     except Exception as e:
          emergency_path = ckpt_mgr._get_filepath(emergency=True)
          torch.save({
               'custom_t5_model_state': custom_t5_model.state_dict(),
               'optimizer_state': optimizer.state_dict(),
               'scheduler_state': scheduler.state_dict(),
               'scaler': scaler.state_dict(),
               'epoch': epoch,
               'global_step': global_step,
               'early_stopper_state': early_stopper.get_state(),
               'exception': str(e)
          }, emergency_path)
          print(f"[Exception] Error!! Checkpoint has saved in {emergency_path}")
          raise e
     
     ## stop monitor
     stop_event.set()
     monitor_thread.join()
     
     best_checkpoint = ckpt_mgr.load_best(device=device)
     if best_checkpoint:
          custom_t5_model.load_state_dict(best_checkpoint['custom_t5_model_state'])
          best_global_step = best_checkpoint.get('global_step', 'N/A')
          print(f"[Checkpoint] Best T5 model state (from epoch {best_checkpoint.get('epoch', 'N/A')}, step {best_global_step}) loaded.")
     else:
          print("[Checkpoint] Best checkpoint not found. Using the model state from the last completed epoch.")
     
     model_fm.save_t5(custom_t5_model)
     
     print_and_save_loss_curve(train_losses, val_losses, early_stopper, label='T5 Fine-tune')
     
     print("--- T5 Fine-tune Finish! ---")
     del custom_t5_model
     del gnn_model
     clean_memory()
