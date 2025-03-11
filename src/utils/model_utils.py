from rouge_score import rouge_scorer
import torch
import gc
from collections import defaultdict
from rouge_score.scoring import Score

def clean_memory():
     if torch.cuda.is_available():
          gc.collect()
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