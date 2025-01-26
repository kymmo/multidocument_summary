import os

from models.gnn_train import train_gnn, get_gnn_trained_embedding

DATA_PATH = os.path.join("..", "data", "multinews")

def model_train(dataset_path):
     # bert_embed_size = 768
     # hidden_size = bert_embed_size
     # out_size = 768 # for long-t5-base input
     # num_heads = 8
     # ## gnn training, t5 freezed
     # train_data_path = os.path.join(DATA_PATH, "train.jsonl")
     # train_gnn(train_data_path, bert_embed_size, bert_embed_size, hidden_size, out_size, num_heads)
     
     # eval_data_path = os.path.join(DATA_PATH, "evaluation.jsonl")
     # gnn_sent_embeddings, node_sent_maps = get_gnn_trained_embedding(eval_data_path, bert_embed_size, bert_embed_size, hidden_size, out_size, num_heads)
     
     # t5 training, gnn freezed
     return
     
     
def freeze_model(model):
     for param in model.parameters():
          param.requires_grad = False

def unfreeze_model(model):
     for param in model.parameters():
          param.requires_grad = True
          
# 将 embedding 投影到 T5 的输入空间
# 这里假设 embedding 已经是 T5 的输入维度
# inputs_embeds = embedding

# # 生成摘要
# summary_ids = model.generate(inputs_embeds=inputs_embeds)

# # 将 token ID 解码为文本
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# print("Summary:", summary)