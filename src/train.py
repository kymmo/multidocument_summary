
from models.gnn_train import train_gnn

def model_train(file_path):
     bert_embed_size = 768
     hidden_size = bert_embed_size
     out_size = 768 # for long-t5-base
     num_heads = 8
     sent_rel_embeddings = train_gnn(file_path, bert_embed_size, bert_embed_size, hidden_size, out_size, num_heads)
     ###TODO torch.save(model.state_dict(), "model.pth")