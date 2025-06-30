import os
from pathlib import Path

from models.model_eval import eval_t5_summary, eval_join_summary
from models.two_stage_train import train_gnn_t5
from models.join_training import run_joint_training

def model_train_eval(dataset_path, learning_rate = 0.001, num_epochs = 20, gnn_batch_size=16, llm_batch_size=4,
                    patience = 5, llm_accumulate_step = 4, gnn_hidden_size = 512, gnn_accumulation_steps=4,
                    sent_similarity_threshold = 0.75, gnn_out_size = 768, num_heads = 8,
                    t5_learning_rates_dict = None, warmup_ratio = 0.1):
     
     train_gnn_t5(
          dataset_path=dataset_path,
          hidden_size=gnn_hidden_size,
          out_size=gnn_out_size,
          num_heads=num_heads,
          learning_rate=learning_rate,
          num_epochs=num_epochs,
          feat_drop=0.1,
          attn_drop=0.1,
          gnn_batch_size=gnn_batch_size,
          llm_batch_size=llm_batch_size,
          patience=patience,
          llm_accumulate_step=llm_accumulate_step,
          gnn_accumulation_steps=gnn_accumulation_steps,
          sent_similarity_threshold=sent_similarity_threshold,
          learning_rates_dict=t5_learning_rates_dict,
          warmup_ratio=warmup_ratio,
     )
     
     eval_data_path = os.path.join(dataset_path, "test.jsonl")
     scores_dict = eval_t5_summary(eval_data_path, max_summary_length = 150, sent_similarity=sent_similarity_threshold)

     return scores_dict

def join_train_eval(dataset_path, gnn_learning_rate = 0.001, num_epochs = 20, batch_size=16,
                    patience = 5, gnn_hidden_size = 512, accumulate_step=4,
                    sent_similarity_threshold = 0.75, gnn_out_size = 768, num_heads = 8,
                    llm_learning_rates_dict = None, warmup_ratio = 0.1,
                    gnn_feat_drop=0.1, gnn_attn_drop=0.1,
                    max_summary_length = 200):
          
     run_joint_training(
          dataset_path=dataset_path,
          gnn_hidden_size=gnn_hidden_size,
          gnn_out_size=gnn_out_size,
          num_heads=num_heads,
          gnn_learning_rate=gnn_learning_rate,
          num_epochs=num_epochs,
          gnn_feat_drop=gnn_feat_drop,
          gnn_attn_drop=gnn_attn_drop,
          batch_size=batch_size,
          patience=patience,
          accumulate_step=accumulate_step,
          sent_similarity_threshold=sent_similarity_threshold,
          llm_learning_rates_dict=llm_learning_rates_dict,
          warmup_ratio=warmup_ratio,
     )
     
     test_data_path = os.path.join(dataset_path, "test.jsonl")
     scores_dict = eval_join_summary(test_data_path, max_summary_length = max_summary_length, sent_similarity=sent_similarity_threshold)

     return scores_dict
