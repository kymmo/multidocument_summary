from rouge_score import rouge_scorer
import torch
import gc
import psutil
from collections import defaultdict
from rouge_score.scoring import Score
import multiprocessing

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
     print(f"[{label}] GPU allocated: {allocated:.2f} GB | preserved: {reserved:.2f} GB")
     
def print_cpu_memory(label, interval = 1):
     memory = psutil.virtual_memory()
     cpu_percent = psutil.cpu_percent(interval=interval)  ## 1 sec interval
     
     print(f"[{label}] CPU used: {cpu_percent}%. Memory used: {memory.percent}%. Memory Detail: {memory.used / (1024 ** 3):.2f} GB | remaining: {memory.available / (1024 ** 3):.2f} GB")
     
def auto_workers():
     """Estimates optimal number of workers based on CPU and memory."""
     try:
          mem_available_gb = psutil.virtual_memory().available / (1024 ** 3)
          # Estimate memory per worker
          model_mem_gb = 1.5 # GB
          cpu_cnt = multiprocessing.cpu_count()

          mem_limit_workers = max(1, int((mem_available_gb * 0.90) / model_mem_gb)) # Use 90% of available mem
          # Leave at least one CPU core free for the main process and OS
          cpu_limit_workers = max(1, cpu_cnt - 1)

          workers = min(cpu_limit_workers, mem_limit_workers)
          return max(workers, 2)
     except Exception as e:
          print(f"[WARN] Failed to determine optimal workers automatically: {e}. Falling back to 2.")
          return 2 # Fallback value