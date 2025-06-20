import torch
import time
from bert_score import score
from torch_geometric.data import Batch
from torch.utils.data import DataLoader as data_DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5TokenizerFast

from utils.model_utils import clean_memory, freeze_model
from models.CustomT5 import reshape_embedding_to_tensors
from models.DatasetLoader import EvalDataset, custom_collate_fn
from models.CheckPointManager import DataType
from models.ModelFileManager import model_fm
from models.two_stage_train import get_combined_embed2
from models.LongTextEncoder import LongTextEncoder
from utils.model_utils import rouge_eval, merge_dicts
from models.InforMetricsCalculator import InforMetricsCalculator

base_model = "google-t5/t5-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_tokenizer = T5TokenizerFast.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained(base_model, use_cache=False).to(device)
t5_model.gradient_checkpointing_enable()

def eval_t5_summary(eval_data_path, max_summary_length, batch_size = 16, sent_similarity = 0.6):
     ## models load
     gnn_model = model_fm.load_gnn()
     gnn_model = gnn_model.to(device)
     gnn_model.eval()
     freeze_model(gnn_model)
     
     fine_tuned_t5 = model_fm.load_t5()
     fine_tuned_t5 = fine_tuned_t5.to(device)
     fine_tuned_t5.eval()
     freeze_model(fine_tuned_t5)
     
     eval_dataset = EvalDataset(file_path=eval_data_path, dataset_type=DataType.TEST.value, sent_similarity=sent_similarity)
     eval_dataloader = data_DataLoader(
          eval_dataset,
          batch_size=batch_size,
          shuffle=False,
          collate_fn=custom_collate_fn
     )
     
     long_text_encoder = LongTextEncoder(t5_tokenizer, t5_model)
     generated_refer_summary_pair_list = []
     original_sents_list = []
     
     print("Start evaluation...")
     try:
          eval_start_time = time.time()
          with torch.no_grad():
               for batch in eval_dataloader:
                    batch_graph, batch_map, batch_summary = batch
                    
                    with torch.cuda.amp.autocast():
                         batched_graph = Batch.from_data_list(batch_graph).to(device, non_blocking=True)
                         with torch.no_grad():
                              sentence_graph_embs, _ = gnn_model(batched_graph)
                              sentence_graph_embs = sentence_graph_embs.detach()
                         
                         sent_texts = batched_graph['sentence'].text
                         sent_text_list = [sent for doc in sent_texts for sent in doc]
                         prompt = "Summarize: "
                         sent_text_list.insert(0, prompt)
                         sentence_text_embs = long_text_encoder.encode_batch(sent_text_list)
                         
                         concat_embs_list = get_combined_embed2(batch_graph, sentence_graph_embs, sentence_text_embs)
                         summaries = generate_t5_summary(fine_tuned_t5, concat_embs_list, max_summary_length)
                         
                         generated_refer_summary_pair_list.append((batch_summary, summaries))
                         original_sents_list.append(sent_texts)
          
          print(f"Finish Summary Generation, time cost:  {time.time() - eval_start_time:.4f} s.")

          rouge_score_dict = get_rouge_score(generated_refer_summary_pair_list=generated_refer_summary_pair_list)
          bert_score = get_bert_score(generated_refer_summary_pair_list=generated_refer_summary_pair_list)
          infor_score = get_infor_score(original_sents_list, generated_refer_summary_pair_list)
          print(f"Finish Evaluation, time cost:  {time.time() - eval_start_time:.4f} s.")

     except Exception as e:
          raise e
     
     
     return {
          'rouge': rouge_score_dict,
          'bert': bert_score,
          "hallucination": infor_score['hallucination'],
          "strong_hallucination": infor_score['strong_hallucination'],
          "faithfulness": infor_score['faithfulness'],
          "omission": infor_score['omission'],
          "contradiction": infor_score['contradiction'],
     }
     
def generate_t5_summary(fine_tuned_t5, combin_embeddings_list, max_summary_length=512):
     with torch.no_grad():
          inputs_comb_embeds, masks = reshape_embedding_to_tensors(combin_embeddings_list)
          inputs_embeds = fine_tuned_t5.projector(inputs_comb_embeds)
          
          generation_config = {
               "max_length": max_summary_length,
               "early_stopping": True,
               "repetition_penalty": 2.0,
               "no_repeat_ngram_size": 3,
               "length_penalty": 1.2,
               "do_sample": False,
               "num_beams": 4,
               "diversity_penalty": 0.7,
               "num_beam_groups": 2,
               "bos_token_id": t5_tokenizer.bos_token_id or t5_tokenizer.pad_token_id,
               "eos_token_id": t5_tokenizer.eos_token_id
          }
          
          outputs = fine_tuned_t5.generate(
               inputs_embeds=inputs_embeds.to(device),
               attention_mask=masks.to(device),
               **generation_config
          )

          decoded_outputs = t5_tokenizer.batch_decode(
               outputs,
               skip_special_tokens=True,
               clean_up_tokenization_spaces=True
          )

          return decoded_outputs
     
def get_rouge_score(generated_refer_summary_pair_list):
     batch_scores = []
     for reference_summary, generated_summary in generated_refer_summary_pair_list:
          score = rouge_eval(reference_list=reference_summary, generated_list=generated_summary)
          batch_scores.append(score)
     
     scores_list = [obj for batch_score in batch_scores for obj in batch_score]
     avg_dict = merge_dicts(scores_list)
     
     return avg_dict

def get_bert_score(generated_refer_summary_pair_list):
     generated_summaries = [summary for pair in generated_refer_summary_pair_list for summary in pair[1]]
     reference_summaries = [summary for pair in generated_refer_summary_pair_list for summary in pair[0]]

     P, R, F1 = score(cands=generated_summaries, refs=reference_summaries, lang='en',
                         model_type='roberta-large', batch_size=64, verbose=False)

     average_f1_score = F1.mean().item()
     
     return round(average_f1_score, 4)

def get_infor_score(original_sents_list, generated_refer_summary_pair_list):
     infor_metrics_cal = InforMetricsCalculator()
     hallucination_rates = []
     faithfulness_scores = []
     omission_rates = []
     strong_hallucinations = []
     contradictions = []
     
     for doc_sents_list, (reference_summary, generated_summary) in zip(original_sents_list, generated_refer_summary_pair_list):
          scores = infor_metrics_cal._get_infor_metrics(doc_sents_list, reference_summary, generated_summary)
          hallucination_rates.append(scores['hallucination'])
          faithfulness_scores.append(scores['faithfulness'])
          omission_rates.append(scores['omission'])
          strong_hallucinations.append(scores['strong_hallucination'])
          contradictions.append(scores['contradiction'])
          
     return {
          "hallucination": round(sum(hallucination_rates) / len(hallucination_rates), 4),
          "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
          "omission": round(sum(omission_rates) / len(omission_rates), 4),
          "strong_hallucination": round(sum(strong_hallucinations) / len(strong_hallucinations), 4),
          "contradiction": round(sum(contradictions) / len(contradictions), 4),
     }