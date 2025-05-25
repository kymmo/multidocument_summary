import os
from pathlib import Path

from models.model_eval import eval_t5_summary
from models.two_stage_train import train_gnn_t5

def model_train_eval(dataset_path, learning_rate = 0.001, num_epochs = 20, gnn_batch_size=16, llm_batch_size=4,
                    patience = 5, llm_accumulate_step = 4,
                    sent_similarity_threshold = 0.75, gnn_out_size = 768, num_heads = 8,
                    t5_learning_rates_dict = None, warmup_ratio = 0.1):
     bert_embed_size = 768
     hidden_size = bert_embed_size
     
     train_gnn_t5(
          dataset_path=dataset_path,
          hidden_size=hidden_size,
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
          sent_similarity_threshold=sent_similarity_threshold,
          learning_rates_dict=t5_learning_rates_dict,
          warmup_ratio=warmup_ratio,
     )
     
     eval_data_path = os.path.join(dataset_path, "test.jsonl")
     scores = eval_t5_summary(eval_data_path, max_summary_length = 150, sent_similarity=sent_similarity_threshold)

     return scores
