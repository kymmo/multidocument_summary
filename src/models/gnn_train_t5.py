import torch
import random
import math
from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as geo_DataLoader
from pytorch_metric_learning import losses
import traceback
from collections import deque
from transformers import get_cosine_schedule_with_warmup

from models.RelHetGraph import RelHetGraph, EdgeKeyTuple
from models.DatasetLoader import OptimizedDataset
from models.CheckPointManager import ModelCheckpointManager, DataType
from models.EarlyStopper import EarlyStopper
from models.ModelFileManager import model_fm
from utils.model_utils import freeze_model, clean_memory, print_gpu_memory, print_and_save_loss_curve

def train_gnn(file_path, hidden_size, out_size, num_heads, val_file_path, sentence_in_size = 768, word_in_size = 768, projection_dim = 768,
               learning_rate=0.001, num_epochs=20, feat_drop=0.2, attn_drop=0.2, batch_size=32, save_method='entire_model', patience=5, sent_similarity_threshold = 0.6,
               gnn_accumulation_steps = 4, warmup_ratio = 0.1,
               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

     clean_memory()

     train_dataset = OptimizedDataset(file_path=file_path, dataset_type=DataType.TRAIN.value, sent_similarity=sent_similarity_threshold)
     train_dataloader = geo_DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=True,
          pin_memory=True,
          num_workers=0,
     )
     
     val_dataset = OptimizedDataset(file_path=val_file_path, dataset_type=DataType.VALIDATION.value, sent_similarity=sent_similarity_threshold)
     val_dataloader = geo_DataLoader(
          val_dataset,
          batch_size=batch_size,
          shuffle=False, # No shuffle for validation
          pin_memory=True,
          num_workers=0,
     )
     
     gnn_model = RelHetGraph(
          hidden_size=hidden_size, 
          out_size=out_size, 
          projection_dim=projection_dim,
          num_heads=num_heads, 
          sentence_in_size=sentence_in_size, 
          word_in_size=word_in_size , 
          document_in_size=sentence_in_size, ## avg of sent embs
          feat_drop=feat_drop, attn_drop=attn_drop).to(device)
     optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)
     
     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gnn_accumulation_steps)
     max_train_steps = num_epochs * num_update_steps_per_epoch
     num_warmup_steps = int(max_train_steps * warmup_ratio)
     print(f"[Scheduler] Total training steps estimated: {max_train_steps}, Warmup steps: {num_warmup_steps}")
     scheduler = get_cosine_schedule_with_warmup(
          optimizer,
          num_warmup_steps=num_warmup_steps,
          num_training_steps=max_train_steps,
          num_cycles=0.5,
          last_epoch=-1,
     )
     scaler = torch.cuda.amp.GradScaler()

     print_gpu_memory("after gnn model loading")
     
     # check point
     ckpt_mgr = ModelCheckpointManager(stage_name="gnn_model")
     early_stopper = EarlyStopper(patience=patience, checkpoint_manager=ckpt_mgr)

     start_epoch = 0
     train_losses = [] # list to store training losses for plotting
     val_losses = []
     if (checkpoint := ckpt_mgr.load(device)) is not None:
          gnn_model.load_state_dict(checkpoint['gnn_model_state'])
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
               print("No EarlyStopper state found in checkpoint. Initializing fresh.")
          
          if 'train_losses' in checkpoint: train_losses = checkpoint['train_losses']
          if 'val_losses' in checkpoint: val_losses = checkpoint['val_losses']
          
          print(f"Resume training! Start from epoch {start_epoch}.")

     last_epoch_completed = -1
     try:
          for epoch in range(start_epoch, num_epochs):
               last_epoch_completed = epoch
               gnn_model.train()
               total_train_loss_epoch  = 0.0
               num_train_batches_processed = 0
               train_empty_batch_cnt = 0
               optimizer.zero_grad()
               
               print('--- GNN Training ---')
               for batch_idx, batch in enumerate(train_dataloader):
                    batch = batch.to(device)
                    
                    with torch.cuda.amp.autocast():
                         _, projected_sent_embeddings = gnn_model(hetero_data=batch, need_projection=True)
                         
                         # contrasive learning
                         s2s_keys = [EdgeKeyTuple.SENT_SIM.value, EdgeKeyTuple.SENT_ANT.value]
                         loss = compute_contractive_learning_loss(
                              projected_sentence_embeddings=projected_sent_embeddings,
                              graph=batch,
                              positive_edge_keys=s2s_keys,
                              temperature = 0.1,
                         )
                    
                         if loss is None:
                              train_empty_batch_cnt += 1
                              continue
                    
                    if torch.is_tensor(loss) and not torch.isnan(loss) and loss.requires_grad:
                         scaled_loss  = loss / gnn_accumulation_steps
                         scaler.scale(scaled_loss).backward()
                         total_train_loss_epoch += loss.item()
                         num_train_batches_processed += 1
                         
                         if (batch_idx + 1) % gnn_accumulation_steps == 0 or (batch_idx + 1) >= len(train_dataloader):
                              # scaler.unscale_(optimizer)
                              # torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=5.0)

                              scaler.step(optimizer)
                              scaler.update()
                              optimizer.zero_grad()
                    else:
                         print(f"[Warning] NaN loss in training batch {batch_idx}. Skipping update.")
               
               avg_train_loss = total_train_loss_epoch / num_train_batches_processed if num_train_batches_processed > 0 else 0
               train_losses.append(avg_train_loss)
               print(f"[Training] Get {train_empty_batch_cnt} / {len(train_dataloader)} EMPTY batch.")
               print(f"[Training] Epoch {epoch + 1} / {num_epochs}, Loss: {avg_train_loss:.4f}, Training Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
               
               
               # --- Validation for Early Stop ---
               print('--- GNN Validation ---')
               gnn_model.eval()
               total_val_loss = 0.0
               num_val_batches_processed = 0
               val_empty_batch_cnt = 0
               with torch.no_grad():
                    for batch_idx_val, batch in enumerate(val_dataloader):
                         batch = batch.to(device)
                         
                         with torch.cuda.amp.autocast():
                              _, projected_sent_embeddings = gnn_model(hetero_data=batch, need_projection=True)
                         
                              # contrasive learning
                              s2s_keys = [EdgeKeyTuple.SENT_SIM.value, EdgeKeyTuple.SENT_ANT.value]
                              val_loss = compute_contractive_learning_loss(
                                   projected_sentence_embeddings=projected_sent_embeddings,
                                   graph=batch,
                                   positive_edge_keys=s2s_keys,
                                   temperature = 0.1,
                              )
                         
                              if val_loss is None:
                                   val_empty_batch_cnt += 1
                                   continue
                              
                         if torch.is_tensor(val_loss) and not torch.isnan(val_loss):
                              total_val_loss += val_loss.item()
                              num_val_batches_processed += 1
                         elif torch.isnan(val_loss):
                              print(f"Warning: NaN loss in validation batch {batch_idx_val}.")
                              
               avg_val_loss = total_val_loss / num_val_batches_processed if num_val_batches_processed > 0 else 0
               val_losses.append(avg_val_loss)
               print(f"[Validation] Get {val_empty_batch_cnt} / {len(val_dataloader)} EMPTY batch.")
               print(f"[Validation] Epoch {epoch + 1}/{num_epochs}, Loss: {avg_val_loss:.4f}")
               
               current_early_stopper_state = early_stopper.get_state()
               scheduler.step(avg_val_loss)
               
               models_to_save = {'gnn_model': gnn_model}
               optimizers_to_save = {'optimizer': optimizer}
               schedulers_to_save = {'scheduler': scheduler}
               ckpt_path = ckpt_mgr.save(
                    epoch=epoch,
                    models=models_to_save,
                    optimizers=optimizers_to_save,
                    schedulers=schedulers_to_save,
                    scaler=scaler,
                    early_stopper_state=current_early_stopper_state,
                    train_losses=train_losses,
                    val_losses=val_losses
               )
               print(f"[checkpoint saved] {epoch}-th epoch checkpoint has saved to path {ckpt_path}")
               
               if early_stopper(val_loss=avg_val_loss, epoch=epoch,
                              models=models_to_save, optimizers=optimizers_to_save,
                              schedulers=schedulers_to_save, scaler=scaler,
                              train_losses_history=train_losses, val_losses_history=val_losses):
                    break # Stop training
               
     except Exception as e:
          current_epoch = last_epoch_completed if last_epoch_completed >= 0 else start_epoch -1
          emergency_path = ckpt_mgr._get_filepath(epoch=current_epoch, emergency=True)
          torch.save({
               'gnn_model_state': gnn_model.state_dict(),
               'optimizer_state': optimizer.state_dict(),
               'scheduler_state': scheduler.state_dict(),
               'scaler': scaler.state_dict(),
               'epoch': current_epoch,
               'early_stopper_state': early_stopper.get_state(),
               'train_losses': train_losses,
               'val_losses': val_losses,
               'exception': str(e)
          }, emergency_path)
          print(f"[Exception] Error!! Checkpoint has saved in {emergency_path}")
          raise e
     
     best_checkpoint = ckpt_mgr.load_best(device=device)
     best_gnn_model = None
     if best_checkpoint:
          best_gnn_model = RelHetGraph(
          hidden_size=hidden_size, 
          out_size=out_size, 
          projection_dim=projection_dim,
          num_heads=num_heads, 
          sentence_in_size=sentence_in_size, 
          word_in_size=word_in_size , 
          document_in_size=sentence_in_size, ## avg of sent embs
          feat_drop=feat_drop, attn_drop=attn_drop).to(device)
          
          best_gnn_model.load_state_dict(best_checkpoint['gnn_model_state'])
          gnn_model = best_gnn_model
          print(f"[Checkpoint] The least-loss GNN model (from epoch {best_checkpoint.get('epoch', 'N/A')}) is reloaded from checkpoint.")
     
     print_and_save_loss_curve(train_losses, val_losses, early_stopper, label='GNN Training')
     
     if save_method == 'entire_model':
          ## save entire model
          model_fm.save_gnn(gnn_model)
     elif save_method == 'weights':
          torch.save(gnn_model.state_dict(), 'gnn_trained_weights.pth')
     else:
          raise ValueError(f'save_method has no value as {save_method}.')
     
     print("--- GNN Training Finish! ---")
     del gnn_model
     if best_gnn_model is not None:
          del best_gnn_model
     del optimizer, scheduler, scaler
     clean_memory()

def compute_contractive_learning_loss(
          projected_sentence_embeddings,
          graph,
          positive_edge_keys,
          temperature = 0.1,
          min_cluster_size_for_positives = 2,
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
     ):
     """
     Computes contrastive loss based on graph connectivity.
     
     Args:
          projected_sentence_embeddings: Embeddings from the GNN's projection head.
          data: The PyG HeteroData object.
          positive_edge_key_tuples: List of edge type tuples defining positive relationships
                                   between 'sentence' nodes.
          temperature: Temperature for the SupConLoss.
          device: Current computation device.
          min_cluster_size_for_positives: Components smaller than this size will have their
                                             nodes treated as distinct singletons.
     Returns:
          torch.Tensor: The computed contrastive loss.
     """
     num_total_sentences = projected_sentence_embeddings.size(0)

     if num_total_sentences == 0 or num_total_sentences < min_cluster_size_for_positives:
          print("[WARN] [Contrasive Learning] No sentence embeddings provided for contrastive loss.")
          return None

     # --- 1. Generate Labels
     graph_labels = torch.full((num_total_sentences,), -1, dtype=torch.long, device=device)
     adj = defaultdict(list)
     for edge_key_tuple in positive_edge_keys:
          if not (isinstance(edge_key_tuple, tuple) and len(edge_key_tuple) == 3 and \
                    edge_key_tuple[0] == 'sentence' and edge_key_tuple[2] == 'sentence'):
               print(f"[WARN] [Contrasive Learning] Skipping non-sentence-to-sentence edge type {edge_key_tuple} for S2S component labeling.")
               continue # only consider sentence-to-sentence positive edges for these labels
          
          if edge_key_tuple not in graph.edge_index_dict or graph.edge_index_dict[edge_key_tuple].numel() == 0:
               continue

          edge_index = graph.edge_index_dict[edge_key_tuple]
          for k in range(edge_index.size(1)): ## originally bidirectional
               u, v = edge_index[0, k].item(), edge_index[1, k].item()
               adj[u].append(v)

     if not adj: ## graph data is empty
          return None
     
     # BFS to find connected components
     current_component_id = 0
     for node_start_idx in range(num_total_sentences):
          if graph_labels[node_start_idx] == -1:
               component_nodes_indices = []
               q = deque([node_start_idx])
               graph_labels[node_start_idx] = current_component_id
               component_nodes_indices.append(node_start_idx)
               while len(q) > 0:
                    curr_u = q.popleft()
                    for curr_v in adj.get(curr_u, []):
                         if graph_labels[curr_v] == -1:
                              graph_labels[curr_v] = current_component_id
                              component_nodes_indices.append(curr_v)
                              q.append(curr_v)
               
               # If a component is too small, re-label its members to be distinct singletons
               if len(component_nodes_indices) < min_cluster_size_for_positives:
                    temp_singleton_id_start = current_component_id
                    for i, node_idx_in_small_component in enumerate(component_nodes_indices):
                         graph_labels[node_idx_in_small_component] = temp_singleton_id_start + i
                    current_component_id = temp_singleton_id_start + len(component_nodes_indices)
               else:
                    current_component_id += 1

     
     # --- 2. handle singletons
     unassigned_mask = (graph_labels == -1)
     if torch.any(unassigned_mask):
          num_unassigned = torch.sum(unassigned_mask).item()
          graph_labels[unassigned_mask] = torch.arange(
               current_component_id,
               current_component_id + num_unassigned,
               device=device, dtype=torch.long
          )

     unique_labels, counts = torch.unique(graph_labels, return_counts=True)
     has_positive_groups = torch.any(counts >= min_cluster_size_for_positives)

     if not has_positive_groups and num_total_sentences > 1 :
          print(f"[WARN] [Contrasive Learning] Contrastive Loss: No label groups large enough (>= {min_cluster_size_for_positives}) to form positive pairs. Skip.")
          return None
     
     if len(unique_labels) == 1 and num_total_sentences > 1: ## only one big group for all sents
          print(f"[WARN] [Contrasive Learning] All {num_total_sentences} sentences assigned the same label ({unique_labels[0]}). Skip.")
          return None
     
     
     # --- 3. Initialize and Compute Loss using pytorch-metric-learning ---
     loss_func = losses.SupConLoss(temperature=temperature)

     try:
          normalized_embeddings = torch.nn.functional.normalize(projected_sentence_embeddings, p=2, dim=1)
          loss = loss_func(normalized_embeddings, graph_labels)
     except Exception as e:
          print(f"[ERROR] [Contrasive Learning] when Contrastive loss computation: {e}")
          traceback.print_exc()
          return torch.tensor(0.0, device=device, requires_grad=False)

     if torch.isnan(loss) or torch.isinf(loss):
          print(f"[WARN] [Contrasive Learning] Contrastive loss is NaN or Inf. Skip.")
          return torch.tensor(0.0, device=device, requires_grad=False)

     return loss
