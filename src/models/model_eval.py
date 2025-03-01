import torch
import time
import torch.nn as nn
from torch.cuda.amp import autocast
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch.utils.data import DataLoader as data_DataLoader

from models.gnn_train_t5 import t5_tokenizer, t5_model, device
from utils.model_utils import clean_memory, freeze_model
from models.CustomT5 import CustomT5, reshape_embedding_to_tensors
from models.RelHetGraph import RelHetGraph
from models.DatasetLoader import EvalDataset, custom_collate_fn
from models.two_stage_train import get_combined_embed
from utils.model_utils import rouge_eval, merge_dicts

def eval_t5_summary(eval_data_path, max_summary_length, batch_size = 16):
     ## models load
     gnn_model = torch.load('gnn_trained_weights.pt')
     gnn_model.eval()
     freeze_model(gnn_model)
     
     fine_tuned_t5 = CustomT5.from_pretrained("./fine_tuned_t5").to(device)
     fine_tuned_t5.eval()
     freeze_model(fine_tuned_t5)
     
     eval_dataset = EvalDataset(eval_data_path)
     eval_dataloader = data_DataLoader(
          eval_dataset,
          batch_size=batch_size,
          shuffle=False,
          pin_memory=True,
          collate_fn=custom_collate_fn
     )
     
     print("Start evaluation...")
     eval_start_time = time.time()
     batch_scores = []
     with torch.no_grad():
          for batch in eval_dataloader:
               batch_graph, batch_map, batch_summary = batch
               
               with torch.cuda.amp.autocast():
                    batched_graph = Batch.from_data_list(batch_graph).to(device, non_blocking=True)
                    sentence_feat = batched_graph['sentence'].x
                    word_feat = batched_graph['word'].x
                    sent_text = batched_graph['sentence'].text
                    
                    corrupted_sentence_feat = F.dropout(sentence_feat, p=0.1, training=gnn_model.training)
                    gnn_embeddings = gnn_model(batched_graph, corrupted_sentence_feat, word_feat)
                    concat_embs = get_combined_embed(batch_graph, gnn_embeddings, sent_text)
                    summaries = generate_t5_summary(fine_tuned_t5, concat_embs, max_summary_length)
                    
                    batch_scores.append(rouge_eval(batch_summary, summaries))
     
     # get average scores
     scores_list = [obj for batch_score in batch_scores for obj in batch_score]
     avg = merge_dicts(scores_list)
     eval_end_time = time.time()
     print(f"Finish evaluation, time cost:  {eval_end_time - eval_start_time:.4f} s.")
     
     return avg


def generate_t5_summary(fine_tuned_t5, combin_embeddings_list, max_summary_length=512):
     with torch.no_grad():
          inputs_comb_embeds, masks = reshape_embedding_to_tensors(combin_embeddings_list)
          inputs_embeds = fine_tuned_t5.projector(inputs_comb_embeds)
          
          generation_config = {
               "max_length": max_summary_length,
               "num_beams": 5,
               "early_stopping": True,
               "repetition_penalty": 2.0,
               "no_repeat_ngram_size": 4,
               "length_penalty": 1.0,
               "temperature": 0.9,
               "do_sample": True,
               "top_k": 50,
               "top_p": 0.95,
               "bos_token_id": t5_tokenizer.pad_token_id,
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


def get_t5_outputs2(gnn_sent_embeddings, sample_node_sent_maps, summary_length = 200, sequence_length = 512):
     gnn_sent_embs = [] ## reset node embedding, to let each map to corresponding node
     start = 0
     for doc_node_sent_map in sample_node_sent_maps:
          sample_sents = gnn_sent_embeddings[start: start + len(doc_node_sent_map)]
          gnn_sent_embs.append(sample_sents)
          start = start + len(doc_node_sent_map)
     
     out_size = gnn_sent_embs[0].shape[1]
     T5_embed_projector = nn.Linear(out_size, t5_model.config.d_model).to(device)
     T5_embed_projector.load_state_dict(
          torch.load('t5_projector_weights.pth', weights_only=True, map_location=device),
          strict=True
     )
                    
     ## process one by one
     generated_summary = [] ## the output string list
     with torch.no_grad(): 
          for embeds, ori_sent_map in zip(gnn_sent_embs, sample_node_sent_maps):
               sent_list = []
               sent_list.append("Generate a summary from documents' embeddings: ")
               sent_list.extend([*ori_sent_map.values()])

               with autocast():
                    ## get T5 embeddings
                    inputs = t5_tokenizer(
                         sent_list, 
                         return_tensors="pt", 
                         padding='max_length',
                         truncation=True, # TODO: deal with long text
                         max_length=sequence_length
                    )
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    t5_model.eval()
                    encoder_sent_outputs = t5_model.encoder(
                              input_ids,
                              attention_mask=attention_mask,
                              return_dict=True,
                    )## encode text infor

                    t5_embeddings = encoder_sent_outputs.last_hidden_state
                    # ignore paddding
                    masked_embeddings = t5_embeddings * attention_mask.unsqueeze(-1)
                    avg_t5_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True) ## (sentence_number, embedding)
                    avg_t5_embeddings = avg_t5_embeddings.to(device)
               
                    ## project GNN embedding to T5 space
                    projected_gnn_embeddings = T5_embed_projector(embeds.to(device))
                    
                    ## combine GNN and T5 embedding
                    task_prefix = avg_t5_embeddings[0].reshape(1, len(avg_t5_embeddings[0]))
                    combined_embeddings = projected_gnn_embeddings + avg_t5_embeddings[1:] ## the first one is task prefix
                    combined_embeddings = torch.cat([task_prefix, combined_embeddings], dim=0)
                    
                    comb_embed_size = combined_embeddings.shape
                    sample_length = comb_embed_size[0]
                    embedding_length = comb_embed_size[1]
                    batch_size = sample_length // sequence_length
                    remaining = sample_length % sequence_length

                    if remaining != 0:
                         padding_size = sequence_length - remaining
                         padding_tensor = torch.zeros(padding_size, embedding_length).to(device)
                         combined_embeddings = torch.cat([combined_embeddings, padding_tensor], dim=0)
                         new_batch_size = batch_size + 1

                         reshaped_tensor = combined_embeddings.view(new_batch_size, sequence_length, embedding_length).to(device)
                         
                         full_mask = torch.ones(batch_size, sequence_length)
                         padding_mask = torch.cat((torch.ones(remaining), torch.zeros(padding_size))).view(1, sequence_length)
                         summary_attention_mask = torch.cat([full_mask, padding_mask], dim=0).to(device)
                    else: ## one sequence represent one sentence
                         reshaped_tensor = combined_embeddings.view(batch_size, sequence_length, embedding_length).to(device)
                         summary_attention_mask = torch.ones(batch_size, sequence_length).to(device)

                    output = t5_model.generate(
                         inputs_embeds=reshaped_tensor,
                         attention_mask=summary_attention_mask,
                         max_length=summary_length,
                         num_beams=3,
                         no_repeat_ngram_size=2,
                         early_stopping=True,
                         use_cache=True,
                         output_scores=True,
                         return_dict_in_generate=True,
                         repetition_penalty=2.5,
                         temperature=0.6,
                         bos_token_id=t5_tokenizer.pad_token_id,
                         eos_token_id=t5_tokenizer.eos_token_id
                    )

                    decoded_output = t5_tokenizer.batch_decode(
                         output[0], 
                         skip_special_tokens=True,
                         clean_up_tokenization_spaces=True
                    )
                    
                    generated_summary.append(" ".join(decoded_output))
                    
                    clean_memory()
     
     return generated_summary
