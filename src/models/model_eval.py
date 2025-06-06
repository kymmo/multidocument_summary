import torch
import time
from torch_geometric.data import Batch
from torch.utils.data import DataLoader as data_DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5TokenizerFast

from utils.model_utils import clean_memory, freeze_model
from models.CustomT5 import CustomT5, reshape_embedding_to_tensors
from models.DatasetLoader import EvalDataset, custom_collate_fn
from models.CheckPointManager import DataType
from models.ModelFileManager import model_fm
from models.two_stage_train import get_combined_embed2
from models.LongTextEncoder import LongTextEncoder
from utils.model_utils import rouge_eval, merge_dicts

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
     
     print("Start evaluation...")
     try: 
          eval_start_time = time.time()
          batch_scores = []
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
                         sentence_text_embs = long_text_encoder.encode_batch(sent_text_list)
                         
                         concat_embs_list = get_combined_embed2(batch_graph, sentence_graph_embs, sentence_text_embs)
                         summaries = generate_t5_summary(fine_tuned_t5, concat_embs_list, max_summary_length)
                         
                         batch_scores.append(rouge_eval(batch_summary, summaries))
          
          # get average scores
          scores_list = [obj for batch_score in batch_scores for obj in batch_score]
          avg = merge_dicts(scores_list)
          eval_end_time = time.time()
          print(f"Finish evaluation, time cost:  {eval_end_time - eval_start_time:.4f} s.")
     
     except Exception as e:
          raise e
     
     return avg

def generate_t5_summary(fine_tuned_t5, combin_embeddings_list, max_summary_length=512):
     with torch.no_grad():
          inputs_comb_embeds, masks = reshape_embedding_to_tensors(combin_embeddings_list)
          inputs_embeds = fine_tuned_t5.projector(inputs_comb_embeds)
          
          generation_config = {
               "max_length": max_summary_length,
               "num_beams": 3,
               "early_stopping": True,
               "repetition_penalty": 2.0,
               "no_repeat_ngram_size": 4,
               "length_penalty": 0.8,
               "do_sample": False,
               # "temperature": 0.9,
               # "top_k": 50,
               # "top_p": 0.95,
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
