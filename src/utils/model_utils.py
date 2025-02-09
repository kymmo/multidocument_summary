from rouge_score import rouge_scorer
import torch
import gc

def clean_memory():
     gc.collect()
     torch.cuda.empty_cache()
     # torch.cuda.set_per_process_memory_fraction(0.8)
     
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