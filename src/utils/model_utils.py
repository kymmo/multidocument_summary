from rouge_score import rouge_scorer
import torch
import gc
import psutil
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from rouge_score.scoring import Score
import multiprocessing

from models.CheckPointManager import parent_path
from models.EmbeddingCompress import AdaptivePoolCompressor

def clean_memory():
     gc.collect()
     if torch.cuda.is_available():
          torch.cuda.empty_cache()
     
def freeze_model(model):
     for param in model.parameters():
          param.requires_grad = False

def unfreeze_model(model):
     for param in model.parameters():
          param.requires_grad = True
          
def rouge_eval(reference_list, generated_list):
     scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
     scores = [scorer.score(ref, gen) for ref, gen in zip(reference_list, generated_list)]

     return scores

def merge_dicts(list_of_dicts):
     averages = defaultdict(lambda: {'precision': [], 'recall': [], 'fmeasure': []})
     
     for d in list_of_dicts:
          for key, score in d.items():
               averages[key]['precision'].append(score.precision)
               averages[key]['recall'].append(score.recall)
               averages[key]['fmeasure'].append(score.fmeasure)
     
     merged = {}
     for key in averages:
          precisions = averages[key]['precision']
          avg_precision = sum(precisions) / len(precisions)
          avg_precision = round(avg_precision, 4)
          
          recalls = averages[key]['recall']
          avg_recall = sum(recalls) / len(recalls)
          avg_recall = round(avg_recall, 4)
          
          fmeasures = averages[key]['fmeasure']
          avg_fmeasure = sum(fmeasures) / len(fmeasures)
          avg_fmeasure = round(avg_fmeasure, 4)
          
          merged_score = Score(
               precision=avg_precision,
               recall=avg_recall,
               fmeasure=avg_fmeasure
          )
          merged[key] = merged_score
     
     return merged

def print_gpu_memory(label):
     allocated = torch.cuda.memory_allocated() / 1024**3
     reserved = torch.cuda.memory_reserved() / 1024**3
     print(f"*[{label}] GPU allocated: {allocated:.2f} GB | preserved: {reserved:.2f} GB")
     
def print_cpu_memory(label, interval = 1):
     memory = psutil.virtual_memory()
     cpu_percent = psutil.cpu_percent(interval=interval)  ## 1 sec interval
     
     print(f"*[{label}] CPU used: {cpu_percent}%. Memory used: {memory.percent}%. Memory Detail: {memory.used / (1024 ** 3):.2f} GB | remaining: {memory.available / (1024 ** 3):.2f} GB")

def monitor_usage(interval, stop_event, label, time_interval = 20):
     """Monitors CPU and memory usage."""
     while not stop_event.wait(time_interval * 60): ## time_interval based on mins
          print_cpu_memory(label, interval)
          print_gpu_memory(label)
          
def auto_workers():
     """Estimates optimal number of workers based on CPU and memory."""
     try:
          mem_available_gb = psutil.virtual_memory().available / (1024 ** 3)
          model_mem_gb = 0.7 # /GB. Estimate memory per worker
          cpu_cnt = multiprocessing.cpu_count()

          mem_limit_workers = max(1, int((mem_available_gb * 0.90) / model_mem_gb)) # Use 90% of available mem
          cpu_limit_workers = max(1, cpu_cnt - 1)

          workers = min(cpu_limit_workers, mem_limit_workers)
          return max(workers, 2)
     except Exception as e:
          print(f"[WARN] Failed to determine optimal workers automatically: {e}. Falling back to 2.")
          return 2 # Fallback value
     
def print_and_save_loss_curve(train_losses, val_losses, early_stopper, label = 'UNKNOWN'):
     SAVE_DIR = os.path.join(parent_path, "images")
     os.makedirs(SAVE_DIR, exist_ok=True)
     SAVE_PATH = os.path.join(SAVE_DIR, f"{label}_loss_curve.png")

     plt.figure(figsize=(10, 6))
     
     if len(train_losses) == 0 or len(val_losses) == 0:
          print("No loss data to plot.")
          return
     
     epochs_ran = list(range(1, len(train_losses) + 1))
     
     if len(epochs_ran) > 1:
          plt.plot(epochs_ran, train_losses, 'b-o', label='Training Loss')
          plt.plot(epochs_ran, val_losses, 'r-x', label='Validation Loss')
     else:
          plt.scatter(epochs_ran, train_losses, c='blue', marker='o', label='Training Loss')
          plt.scatter(epochs_ran, val_losses, c='red', marker='x', label='Validation Loss')
     

     if early_stopper.early_stop_triggered:
          stopped_epoch_for_plot = early_stopper.stopped_epoch + 1
          
          if 0 <= early_stopper.stopped_epoch < len(val_losses):
               plt.scatter(
                    stopped_epoch_for_plot,
                    val_losses[early_stopper.stopped_epoch],
                    color='green',
                    marker='s',
                    s=150,
                    zorder=10,
                    label=f'Early Stop @ Epoch {stopped_epoch_for_plot}'
               )
               plt.axvline(
                    x=stopped_epoch_for_plot,
                    color='gray',
                    linestyle='--',
                    linewidth=1,
                    alpha=0.7
               )
     
     plt.title(f'{label} Loss')
     plt.xlabel('Epoch')
     plt.ylabel('Loss')
     plt.legend()
     plt.grid(True, alpha=0.3)
     
          
     try:
          plt.savefig(SAVE_PATH, bbox_inches='tight', dpi=300, format="png")
          print(f"Loss curve plot saved to {SAVE_PATH}")
          
          plt.show()
     except Exception as plot_e:
          print(f"Error saving plot: {plot_e}")
     finally:
          plt.close()
          
def reshape_embedding_to_tensors(combin_embeddings_list, device, max_len = 512):
     processed_embeddings_list = [emb.to(device) for emb in combin_embeddings_list]
     
     reshape_list = [] ##[tensor(1, sequence_size, embed_size)]
     masks = [] ## (batch_size, sequence_size)
     
     max_node_num = max(graph_embs.shape[0] for graph_embs in processed_embeddings_list)
     max_node_num = min(max_len, max_node_num)
     for graph_embs in processed_embeddings_list:
          cur_len = graph_embs.shape[0]
          
          padding_size = max_node_num - cur_len
          if padding_size > 0:
               graph_embs = torch.cat([
                    graph_embs,
                    torch.zeros(padding_size, graph_embs.shape[1], device=device)
               ], dim=0)
          elif padding_size < 0:
               graph_embs = graph_embs[:max_node_num, :]
               cur_len = max_node_num
               
          reshape_list.append(graph_embs)
          
          mask = torch.zeros(max_node_num, device=device)
          mask_len = cur_len if cur_len <= max_len else max_len
          mask[:mask_len] = 1
          masks.append(mask)
          
     return torch.stack(reshape_list), torch.stack(masks)

def adapt_embeddings(batch_token_list, emb_dim, device, max_len = 512):
     compressor = AdaptivePoolCompressor(emb_dim=emb_dim, target_len=max_len).to(device)
     
     flatten_emb = [] ## (batch_graph_size, graph_token_size, emb_dim)
     for graph_tokens in batch_token_list:
          graph_embs = []
          
          for sent_tokens in graph_tokens:
               graph_embs.extend(sent_tokens)
                    
          flatten_emb.append(torch.cat(graph_embs, dim=0).to(device))
     
     adapt_len_embs = []
     masks = []
     for batch_tok_embs in flatten_emb: ## [token_len, emb_dim]
          cur_len = batch_tok_embs.shape[0]
          if cur_len > max_len:
               compressed = compressor(batch_tok_embs.unsqueeze(0))
               cur_emb = compressed.squeeze(0)
          elif cur_len < max_len: # padding
               padding = torch.zeros((max_len - cur_len), batch_tok_embs.shape[1], device=device)
               cur_emb = torch.concat([batch_tok_embs, padding], dim=0)
          else:
               cur_emb = batch_tok_embs
               
          adapt_len_embs.append(cur_emb)
          
          mask = torch.zeros(max_len)
          mask_len = cur_len if cur_len <= max_len else max_len
          mask[:mask_len] = 1
          masks.append(mask)
          
     return torch.stack(adapt_len_embs, dim=0).to(device), torch.stack(masks, dim=0).to(device)