import networkx as nx
import torch
from transformers import BertTokenizer, BertModel, BertConfig


from utils.data_preprocess_utils import define_node_edge

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def create_graph(word_nodeId_list, sent_nodeId_list, edge_data_list):
     graph_list = []
     
     for (word_node_map, sent_node_map, edges_data) in zip(word_nodeId_list, sent_nodeId_list, edge_data_list):
          ## create graph for each multi-doc training sample
          graph = nx.MultiDiGraph()
          
          for word,w_node_id in word_node_map.items():
               graph.add_node(w_node_id, type = "word", text = word)
          
          for sent,s_node_id in sent_node_map.items():
               graph.add_node(s_node_id, type = "sentence", text = sent)
     
          for (node1, node2), edges in edges_data.items():
               for edge in edges:
                    graph.add_edge(node1, node2, edge_type=edge['type'], weight=edge['weight'] )
                    graph.add_edge(node2, node1, edge_type=edge['type'], weight=edge['weight'] )

          graph_list.append(graph)
     
     return graph_list

def embed_nodes(graphs, sentid_node_map_list):
     """Embeds nodes in the graph using SBERT, BERT, and positional embeddings."""
     embedded_graphs = []
     sent_node_embedding_map_list = get_sent_positional_encoding(sentid_node_map_list)
     
     for (graph, sent_node_embedding_map) in zip(graphs, sent_node_embedding_map_list):
          for node, data in graph.nodes(data = True):
               if data["type"] == "sentence":
                    with torch.no_grad():
                         graph.nodes[node]['embedding'] = sent_node_embedding_map[node]
               if data["type"] == "word":
                    with torch.no_grad():
                         bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                         bert_model = BertModel.from_pretrained('bert-base-uncased')
                         word_embedding_layer = bert_model.embeddings.word_embeddings
                         word_tokens = bert_tokenizer(graph.nodes[node]['text'], return_tensors='pt')
                         token = word_embedding_layer(word_tokens['input_ids'][0])
                         word_embedding = torch.mean(token, dim=0)
                         
                         graph.nodes[node]['embedding'] = word_embedding

          embedded_graphs.append(graph)
     return embedded_graphs


def get_sent_positional_encoding(sentid_node_map_list):
     sent_pos_emb_list = []
     
     for sentid_node_map in sentid_node_map_list:
          sent_node_emb_map = {}
          doc_sentinel = -1
          doc_size = 1 + next(reversed(sentid_node_map.items()))[0][1] ## the second key is doc_id
          # doc absolute position embedding
          bert_abs_config = BertConfig.from_pretrained("bert-base-uncased")
          bert_abs_config.position_embedding_type = "absolute"
          bert_abs_model = BertModel(bert_abs_config)
          doc_input_ids = [i for i in range(doc_size)]
          doc_pos_embeddings = bert_abs_model.embeddings.position_embeddings(torch.tensor(doc_input_ids))
          sent_pos_embeddings = []

          for (training_id, doc_id, sent_id), node_id in reversed(sentid_node_map.items()):
               if(doc_id == doc_sentinel):
                    node_embedding = doc_pos_embeddings[doc_id] + sent_pos_embeddings[sent_id]
                    sent_node_emb_map[node_id] = node_embedding
                    continue
               
               doc_sentinel = doc_id
               
               # sentence relative position embedding
               bert_relative_config = BertConfig.from_pretrained("bert-base-uncased")
               bert_relative_config.position_embedding_type = "relative_key"
               bert_relative_model = BertModel(bert_relative_config)
               sent_input_ids = [i for i in range(sent_id + 1)]
               sent_pos_embeddings = bert_relative_model.embeddings.position_embeddings(torch.tensor(sent_input_ids))
               
               node_embedding = doc_pos_embeddings[doc_id] + sent_pos_embeddings[sent_id]
               sent_node_emb_map[node_id] = node_embedding

          sent_pos_emb_list.append(sent_node_emb_map)
     
     return sent_pos_emb_list