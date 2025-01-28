import os
import torch

from models.gnn_train_t5 import train_gnn, get_gnn_trained_embedding
from models.t5 import get_t5_outputs
from utils.model_utils import rouge_eval

def model_train_and_eval_t5(dataset_path):
     bert_embed_size = 768
     hidden_size = bert_embed_size
     out_size = 768 # for t5-base input
     num_heads = 8
     ## gnn training, t5 freezed
     print(f"Start training GNN. Parameters: hidden_size: {hidden_size}, out_size: {out_size}, attention_heads: {num_heads}")
     train_data_path = os.path.join(dataset_path, "train.jsonl")
     train_gnn(
          file_path=train_data_path,
          hidden_size=hidden_size,
          out_size = out_size, 
          num_heads = num_heads,
          sentence_in_size = 768, 
          word_in_size = 768,
          learning_rate=0.001, 
          num_epochs=20,
          feat_drop=0.2, 
          attn_drop=0.2, 
          batch_size=32
     )
     
     eval_data_path = os.path.join(dataset_path, "evaluation.jsonl")
     gnn_output_embeddings, merged_node_map_list, merged_summary_list = get_gnn_trained_embedding(eval_data_path, hidden_size, out_size, num_heads,sentence_in_size = 768, word_in_size = 768, feat_drop=0.2, attn_drop=0.2, batch_size=32)
     generated_summary = get_t5_outputs(gnn_sent_embeddings=gnn_output_embeddings, sample_node_sent_maps=merged_node_map_list)
     scores = rouge_eval(merged_summary_list, generated_summary)

     print("Finish. scores: ", scores)
     return scores
     