import torch
import math
import os
from pathlib import Path
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from transformers import T5Config, T5TokenizerFast, T5ForConditionalGeneration, T5Tokenizer, get_cosine_schedule_with_warmup

from models.DatasetLoader import JointTrainingDataset, joint_collate_fn
from models.JoinModel import JointOrchestrator, JointOrchestratorwithPrefix
from models.CheckPointManager import ModelCheckpointManager, DataType
from models.EarlyStopper import EarlyStopper
from utils.model_utils import clean_memory, print_and_save_loss_curve
from models.ModelFileManager import model_fm

def run_joint_training(
     dataset_path, gnn_hidden_size, gnn_out_size, num_heads=8, gnn_learning_rate=0.001, num_epochs=20, 
     gnn_feat_drop=0.1, gnn_attn_drop=0.1, batch_size = 16, prefix_len = 10, encoder_learning_rate=3e-4,
     patience=5, accumulate_step=4, sent_similarity_threshold=0.6,
     llm_learning_rates_dict = None, warmup_ratio=0.1
):

     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     base_model_name = "t5-base"
     # base_model_name = "t5-small"
     train_data_path = os.path.join(dataset_path, "train.jsonl")
     val_data_path = os.path.join(dataset_path, "validation.jsonl")
     if not Path(train_data_path).exists() or not Path(val_data_path).exists():
          raise FileNotFoundError(f"File path {train_data_path} or {val_data_path} is not exist!")
          
     if llm_learning_rates_dict is None:
          llm_learning_rates_dict = {
               "shallow_layers": 2e-5,
               "deep_layers": 5e-6,
          }
     
     train_dataset = JointTrainingDataset(file_path=train_data_path, dataset_type=DataType.TRAIN.value, sent_similarity=sent_similarity_threshold)
     train_dataloader = DataLoader(
          train_dataset, 
          batch_size=batch_size, 
          shuffle=True, 
          num_workers=0,
          pin_memory=True,
          collate_fn=joint_collate_fn
     )
     
     val_dataset = JointTrainingDataset(file_path=val_data_path, dataset_type=DataType.VALIDATION.value, sent_similarity=sent_similarity_threshold)
     val_dataloader = DataLoader(
          val_dataset, 
          batch_size=batch_size, 
          shuffle=False, # No shuffle
          num_workers=0,
          pin_memory=True,
          collate_fn=joint_collate_fn
     )

     t5_tokenizer = T5Tokenizer.from_pretrained(base_model_name, legacy=True)
     
     gnn_config = {
          'hidden_size': gnn_hidden_size, 
          'out_size': gnn_out_size, 
          'projection_dim': 768,
          'num_heads': num_heads, 
          'sentence_in_size': 768, 
          'word_in_size': 768 , 
          'document_in_size': 768, ## avg of sent embs
          'feat_drop': gnn_feat_drop, 
          'attn_drop': gnn_attn_drop,
     }
     
     orchestrator_model = JointOrchestratorwithPrefix(
          gnn_config=gnn_config, 
          t5_model_name=base_model_name,
          prefix_length=prefix_len, 
          t5_tokenizer=t5_tokenizer,
     ).to(device)

     optimizer_grouped_parameters = [
          {"params": orchestrator_model.gnn.parameters(), "lr": gnn_learning_rate},
          {"params": orchestrator_model.prefix_encoder.parameters(), "lr": encoder_learning_rate},
          {"params": orchestrator_model.custom_t5.encoder.block[-2:].parameters(), "lr": llm_learning_rates_dict["shallow_layers"]},
          {"params": orchestrator_model.custom_t5.decoder.block[-2:].parameters(), "lr": llm_learning_rates_dict["shallow_layers"]},
     ]
     optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accumulate_step)
     max_train_steps = num_epochs * num_update_steps_per_epoch
     scheduler = get_cosine_schedule_with_warmup(
          optimizer, 
          num_warmup_steps=int(max_train_steps * warmup_ratio), 
          num_training_steps=max_train_steps,
          num_cycles=1.0,
          last_epoch=-1,
     )
     scaler = torch.cuda.amp.GradScaler(init_scale=1024)
     
     ckpt_mgr = ModelCheckpointManager(stage_name="orchestrator_model")
     early_stopper = EarlyStopper(patience=patience, min_delta= 0.001, checkpoint_manager=ckpt_mgr)
     resume = False
     start_epoch = 0
     train_losses = [] # list to store training losses for plotting
     val_losses = []
     if (checkpoint := ckpt_mgr.load(device)) is not None:
          orchestrator_model.load_state_dict(checkpoint['orchestrator_model_state'])
          optimizer.load_state_dict(checkpoint['optimizer_state'])
          if 'scheduler_state' in checkpoint:
               scheduler.load_state_dict(checkpoint['scheduler_state'])
          if 'scaler' in checkpoint:
               scaler.load_state_dict(checkpoint['scaler'])
          
          start_epoch = checkpoint['epoch'] + 1
          resume = True

          if 'early_stopper_state' in checkpoint and checkpoint['early_stopper_state']:
               early_stopper.load_state(checkpoint['early_stopper_state'])
          else:
               print("No EarlyStopper state found in checkpoint. Initializing fresh.")
          
          if 'train_losses' in checkpoint: train_losses = checkpoint['train_losses']
          if 'val_losses' in checkpoint: val_losses = checkpoint['val_losses']
          
          print(f"Resume training! From epoch {start_epoch}.")
     
     if early_stopper.early_stop_triggered:
          print("Early stop has been triggered. Skip training.")
          start_epoch = num_epochs
     
     
     for epoch in range(start_epoch, num_epochs):
          print(f"--- Join Training ---")
          orchestrator_model.train()
          total_loss = 0.0
          processed_batches_this_epoch = 0
          optimizer.zero_grad()
          
          for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}"):
               # with torch.cuda.amp.autocast():
               outputs = orchestrator_model(
                    source_text_list = batch['sample_text_list'],
                    batched_graph = batch['batched_graph'].to(device),
                    label_summaries = batch['label_summaries'],
                    cov_lambda=adjust_cov_lambda(epoch, num_epochs),
               )
               loss = outputs.loss
               
               if torch.isnan(loss):
                    print(f"[Warning] NaN loss at batch {batch_idx}. Skipping update.")
                    continue

               scaled_loss = loss / accumulate_step

               scaler.scale(scaled_loss).backward()
               total_loss += loss.item()
               processed_batches_this_epoch += 1

               if (batch_idx + 1) % accumulate_step == 0 or (batch_idx + 1) == len(train_dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    
          avg_train_loss = total_loss / processed_batches_this_epoch if processed_batches_this_epoch > 0 else 0
          train_losses.append(avg_train_loss)
          current_lrs = scheduler.get_last_lr()
          lr_strings = [f"{lr:.8f}" for lr in current_lrs]
          print(f"[Training] Epoch {epoch+1} / {num_epochs}, Loss: {avg_train_loss:.4f}, Learning Rates: {lr_strings}")
          
          print('--- Validation ---')
          orchestrator_model.eval()
          total_val_loss = 0.0
          num_val_batches = 0
          with torch.no_grad():
               for val_batch_id, val_batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {epoch}"):
                    # with torch.cuda.amp.autocast():
                    val_outputs = orchestrator_model(
                         source_text_list = val_batch['sample_text_list'],
                         batched_graph = val_batch['batched_graph'].to(device),
                         label_summaries = val_batch['label_summaries'],
                         cov_lambda=adjust_cov_lambda(epoch, num_epochs),
                    )

                    total_val_loss += val_outputs.loss.item()
                    num_val_batches += 1
                         
          avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
          val_losses.append(avg_val_loss)
          print(f"[Validation] Epoch {epoch+1}/{num_epochs}, Loss: {avg_val_loss:.4f}")

          # --- Early Stopping Check ---
          models_to_save = {'orchestrator_model': orchestrator_model}
          optimizers_to_save = {'optimizer': optimizer}
          schedulers_to_save = {'scheduler': scheduler}
          
          early_stop_check = early_stopper(avg_val_loss, epoch, models_to_save, optimizers_to_save, 
                         schedulers_to_save, scaler,
                         train_losses_history=train_losses, val_losses_history=val_losses)
          
          ckpt_path = ckpt_mgr.save(
               epoch=epoch,
               models=models_to_save,
               optimizers=optimizers_to_save,
               schedulers=schedulers_to_save,
               scaler=scaler,
               early_stopper_state=early_stopper.get_state(),
               train_losses=train_losses,
               val_losses=val_losses
          )
          print(f"[checkpoint saved] {epoch}-th epoch checkpoint has saved to path {ckpt_path}")
          
          if early_stop_check:
               break # Stop training
          
          if resume and epoch == start_epoch:
               resume = False
     
     best_checkpoint = ckpt_mgr.load_best(device=device)
     if best_checkpoint:
          final_model = JointOrchestratorwithPrefix(
               gnn_config=gnn_config, 
               t5_model_name= base_model_name,
               prefix_length=prefix_len, 
               t5_tokenizer=t5_tokenizer,
          ).to(device)
          final_model.load_state_dict(best_checkpoint['orchestrator_model_state'])
          orchestrator_model = final_model
          print(f"[Checkpoint] Best model state (from epoch {best_checkpoint.get('epoch', 'N/A')}).")
     else:
          print("[Checkpoint] Best checkpoint not found. Using the model state from the last completed epoch.")
     
     model_fm.save_join_model(final_model)
     
     print_and_save_loss_curve(train_losses, val_losses, early_stopper, label='Join Training')
     
     print("--- Joint Training Finished! ---")
     del orchestrator_model
     clean_memory()
     
def adjust_cov_lambda(epoch, total_epochs):
     if total_epochs <= 4:
          return 0.2
          
     if epoch < total_epochs // 4:
          return 0.2
     elif epoch < total_epochs // 2:
          return 0.4
     else:
          return 0.5