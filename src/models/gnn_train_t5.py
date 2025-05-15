import torch
import random
import math
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as geo_DataLoader

from models.RelHetGraph import RelHetGraph
from models.DatasetLoader import EvalDataset, OptimizedDataset, custom_collate_fn
from models.CheckPointManager import ModelCheckpointManager
from models.EarlyStopper import EarlyStopper
from models.CheckPointManager import DataCheckpointManager
from models.ModelFileManager import model_fm
from utils.model_utils import freeze_model, clean_memory, print_gpu_memory
from models.LinkPredictor import LinkPredictor


def train_gnn(file_path, hidden_size, out_size, num_heads, val_file_path, sentence_in_size = 768, word_in_size = 768, 
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
          shuffle=False, ##########test!
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
     
     gnn_model = RelHetGraph(hidden_size, out_size, num_heads, sentence_in_size, word_in_size , feat_drop, attn_drop).to(device)
     link_predictor = LinkPredictor(out_size, hidden_neurons_scale_factor=0.5, dropout_rate=0.1).to(device)
     optimizer = torch.optim.Adam(
          list(gnn_model.parameters()) + list(link_predictor.parameters()), 
          lr=learning_rate)
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
     scaler = torch.cuda.amp.GradScaler()

     print_gpu_memory("after gnn model loading")
     
     # check point
     ckpt_mgr = ModelCheckpointManager(stage_name="gnn_model")
     early_stopper = EarlyStopper(patience=patience, checkpoint_manager=ckpt_mgr)

     start_epoch = 0
     if (checkpoint := ckpt_mgr.load(device)) is not None:
          gnn_model.load_state_dict(checkpoint['gnn_model_state'])
          link_predictor.load_state_dict(checkpoint['link_predictor_state'])
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

     last_epoch_completed = -1
     print(f"--- Training ---")
     try:
          for epoch in range(start_epoch, num_epochs):
               last_epoch_completed = epoch
               gnn_model.train()
               link_predictor.train()
               total_loss = 0
               for batch in train_dataloader:
                    batch = batch.to(device)
                    
                    ##############test
                    print("batch", batch)
                    ####################3
                    masked_graph = get_masked_graph(batch, k=2)
                    
                    if masked_graph is None:
                         continue
                    
                    with torch.cuda.amp.autocast():
                         sentence_feat = masked_graph['sentence'].x
                         word_feat = masked_graph['word'].x
                         corrupted_sentence_feat = F.dropout(sentence_feat, p=0.1, training=gnn_model.training)
                         optimizer.zero_grad()
                         sentence_embeddings = gnn_model(masked_graph, corrupted_sentence_feat, word_feat)
                         
                         # predicted similarity edge
                         pos_pairs_inds = masked_graph[('sentence', 'pos_sim_edge', 'sentence')].edge_index
                         neg_pairs_inds = masked_graph[('sentence', 'neg_sim_edge', 'sentence')].edge_index
                         pos_logits = link_predictor(sentence_embeddings[pos_pairs_inds[0]], sentence_embeddings[pos_pairs_inds[1]])
                         neg_logits = link_predictor(sentence_embeddings[neg_pairs_inds[0]], sentence_embeddings[neg_pairs_inds[1]])
                         all_link_logits = torch.cat([pos_logits, neg_logits], dim=0)

                         # labels
                         pos_labels = torch.ones_like(pos_logits)
                         neg_labels = torch.zeros_like(neg_logits)
                         all_labels = torch.cat([pos_labels, neg_labels], dim=0)

                         loss = F.binary_cross_entropy_with_logits(all_link_logits, all_labels)
                         
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
               
               avg_train_loss = total_loss / len(train_dataloader)
               print(f"[Training] Epoch {epoch + 1} / {num_epochs}, Loss: {avg_train_loss:.4f}, Training Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
               
               # --- Validation for Early Stop ---
               print('--- Validation ---')
               gnn_model.eval()
               link_predictor.eval()
               total_val_loss = 0
               with torch.no_grad():
                    for batch in val_dataloader:
                         batch = batch.to(device)
                         masked_graph = get_masked_graph(batch, k=2)
                         
                         if masked_graph is None:
                              continue
                         
                         with torch.cuda.amp.autocast():
                              sentence_feat = masked_graph['sentence'].x
                              word_feat = masked_graph['word'].x
                              sentence_embeddings = gnn_model(masked_graph, sentence_feat, word_feat)
                              
                              # predicted similarity edge
                              pos_pairs_inds = masked_graph[('sentence', 'pos_sim_edge', 'sentence')].edge_index
                              neg_pairs_inds = masked_graph[('sentence', 'neg_sim_edge', 'sentence')].edge_index
                              pos_logits = link_predictor(sentence_embeddings[pos_pairs_inds[0]], sentence_embeddings[pos_pairs_inds[1]])
                              neg_logits = link_predictor(sentence_embeddings[neg_pairs_inds[0]], sentence_embeddings[neg_pairs_inds[1]])
                              all_link_logits = torch.cat([pos_logits, neg_logits], dim=0)

                              # labels
                              pos_labels = torch.ones_like(pos_logits)
                              neg_labels = torch.zeros_like(neg_logits)
                              all_labels = torch.cat([pos_labels, neg_labels], dim=0)

                         loss = F.binary_cross_entropy_with_logits(all_link_logits, all_labels)
                         total_val_loss += loss.item()
                              
               avg_val_loss = total_val_loss / len(val_dataloader)
               print(f"[Validation] Epoch {epoch + 1}/{num_epochs}, Loss: {avg_val_loss:.4f}")
     
               models_to_save = {'gnn_model': gnn_model, 'link_predictor': link_predictor}
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
               'link_predictor_state': link_predictor.state_dict(),
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
          link_predictor.load_state_dict(best_checkpoint['link_predictor_state'])
          gnn_model = best_gnn_model
          print(f"[Checkpoint] The least-loss GNN model (from epoch {best_checkpoint.get('epoch', 'N/A')}) is reloaded from checkpoint.")
     
     if save_method == 'entire_model':
          ## save entire model
          model_fm.save_gnn(gnn_model)
     elif save_method == 'weights':
          torch.save(gnn_model.state_dict(), 'gnn_trained_weights.pth')
     else:
          raise ValueError(f'save_method has no value as {save_method}.')
     
     print("--- GNN Training Finish! ---")
     del gnn_model, best_gnn_model
     del optimizer, scheduler, scaler
     clean_memory()

def get_masked_graph(pyg_graph, k = 2):
     """mask the original graph for link prediction training for GNN model.
     Args:
          pyg_graph (torch_geometric.data.Data): The original graph.
          k (int): The number of negative edges to generate for each positive edge.
     """
     masked_graph = pyg_graph.clone()
     MIN_POSITIVE_LINKS_THRESHOLD = 5

     # --- 1. produce positive and negative edge pairs ---
     pos_ind = None
     if ('sentence', 'similarity', 'sentence') in masked_graph.edge_types:
          pos_ind = masked_graph['sentence', 'similarity', 'sentence'].edge_index
     else:
          print("No positive similarity pairs found in the batch.")
          return None
     
     pos_pairs = set()
     for i in range(pos_ind.size(1)):
          from_node = pos_ind[0][i].item()
          to_node = pos_ind[1][i].item()
          to_add = (from_node, to_node) if from_node < to_node else (to_node, from_node) ## remove bidirection
          pos_pairs.add(to_add)
     
     if pos_pairs is None or len(pos_pairs) == 0:
          print("No positive similarity pairs found in the batch.")
          return None
     
     if len(pos_pairs) < MIN_POSITIVE_LINKS_THRESHOLD:
          print("Not enough positive similarity pairs found in the graph.")
          return None
     
     neg_pairs = set()
     sent_num = masked_graph['sentence'].x.size(0)
     max_neg_pairs = min(len(pos_pairs) * k, (math.comb(sent_num, 2) - len(pos_pairs)))
     while len(neg_pairs) < max_neg_pairs:
          node1 = random.randint(0, sent_num - 1)
          node2 = random.randint(0, sent_num - 1)
          if node1 == node2:
               continue
          
          to_add = (node1, node2) if node1 < node2 else (node2, node1)
          if to_add not in pos_pairs:
               neg_pairs.add(to_add)
     
     if neg_pairs is None or len(neg_pairs) == 0:
          raise ValueError("No negative similarity pairs found in the graph.")
     
     pos_pairs = list(map(list, pos_pairs))
     neg_pairs = list(map(list, neg_pairs))
     masked_graph[('sentence', 'pos_sim_edge', 'sentence')].edge_index = torch.tensor(pos_pairs, dtype=int).t()
     masked_graph[('sentence', 'neg_sim_edge', 'sentence')].edge_index = torch.tensor(neg_pairs, dtype=int).t()
     
     # --- 2. remove similarity edge ---
     if ('sentence', 'similarity', 'sentence') in masked_graph.edge_types:
          del masked_graph['sentence', 'similarity', 'sentence']
     
     return masked_graph
