import networkx as nx
import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel


from utils.data_preprocess_utils import define_node_edge

sbert_model = SentenceTransformer('all-mpnet-base-v2')
bert_bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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

def embed_nodes(graphs, d_model = 768):
     """Embeds nodes in the graph using SBERT, BERT, and positional embeddings."""
     embedded_graphs = []

     for graph in graphs:
          max_sentence_index = max([node for node, data in graph.nodes(data = True) if data["type"] == "sentence"], default=0)
          max_num_doc = len(graphs)
          pe_sent = get_positional_encoding(max_sentence_index + 1 , d_model)
          
          for node, data in graph.nodes(data = True):
               if data["type"] == "sentence": ###NO NEED
                    with torch.no_grad():
                         sentence_embedding = sbert_model.encode(data["text"], convert_to_tensor = True)
                         graph.nodes[node]['embedding'] = torch.cat((sentence_embedding, pe_sent[node]), dim = -1) # concat with positional embedding
               elif data["type"] == "word":
                    with torch.no_grad():
                         inputs = bert_tokenizer(data["text"], return_tensors = "pt", truncation = True, padding = True)
                         outputs = bert_model(**inputs)
                         word_embedding = outputs.last_hidden_state.mean(dim = 1).squeeze()
                         graph.nodes[node]['embedding'] = word_embedding

          embedded_graphs.append(graph)
     return embedded_graphs