import networkx as nx
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
# from sklearn.decomposition import PCA

from utils.data_preprocess_utils import define_node_edge, load_jsonl

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
sentBERT_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embed_graph(file_path):
     docs_list, summary_list = load_jsonl(file_path)
     sample_graphs, node_maps = create_embed_graphs(docs_list)
     
     return sample_graphs

def get_embed_graph_node_map(file_path):
     docs_list, summary_list = load_jsonl(file_path)
     sample_graphs, node_maps = create_embed_graphs(docs_list)
     
     return sample_graphs, node_maps, summary_list

def create_embed_graphs(docs_list, sent_similarity = 0.6):
     word_nodeId_list, sent_nodeId_list, edge_data_list, sentid_node_map_list = define_node_edge(docs_list, sent_similarity)
     graph_list = create_graph(word_nodeId_list, sent_nodeId_list, edge_data_list)
     embedded_graph_list = embed_nodes(graph_list, sentid_node_map_list)
     pyg_graph_list, nodeid_to_sent_map_list = convert_graph_from_nx_to_pyg(embedded_graph_list)
     node_sent_map_list = get_node_sent_map(embedded_graph_list, nodeid_to_sent_map_list) ## id -> sent
     
     return pyg_graph_list, node_sent_map_list

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
     sent_node_embedding_map_list = get_sent_pos_encoding(sentid_node_map_list)
     
     for (graph, sent_node_embedding_map) in zip(graphs, sent_node_embedding_map_list):
          for node, data in graph.nodes(data = True):
               if data["type"] == "sentence":
                    with torch.no_grad():
                         positon_embeddings = sent_node_embedding_map[node].detach() ## (768)
                         sent_text = graph.nodes[node]['text'][2]
                         sent_embeddings = sentBERT_model.encode(sent_text) ##(384, )
                         sent_embeddings = torch.from_numpy(sent_embeddings)
                         ## mean value padding
                         mean_value = sent_embeddings.mean()
                         padding = torch.full((384,), mean_value) 
                         combined_emb = torch.cat([sent_embeddings, padding], dim=0)
                         
                         graph.nodes[node]['embedding'] = positon_embeddings + combined_emb
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

def get_sent_pos_encoding(sentid_node_map_list):
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

def convert_graph_from_nx_to_pyg(graphs):
     hetro_graphs = []
     nodeid_to_sent_txt_map_list = []
     
     for nx_graph in graphs:
          het_graph = HeteroData()
          node_map = {}
          
          sent_id = 0
          word_id = 0
          word_nodes = []
          sent_nodes = []
          for node in nx_graph.nodes():
               cur_type = nx_graph.nodes[node]['type']
               embed = nx_graph.nodes[node]['embedding']
               if cur_type == 'word':
                    word_nodes.append(embed)
                    node_map[node] = ('word', word_id)
                    word_id = word_id + 1
               elif cur_type == 'sentence':
                    sent_nodes.append(embed)
                    node_map[node] = ('sentence', sent_id)
                    sent_id = sent_id + 1
          het_graph['word'].x = torch.stack(word_nodes)
          het_graph['sentence'].x = torch.stack(sent_nodes)
          
          similarity_edge_indices = []
          similarity_edge_attrs = []
          pro_ant_edge_indices = []
          pro_ant_edge_attrs = []
          sent_word_edge_indices = []
          sent_word_edge_attrs = []
          word_sent_edge_indices = []
          word_sent_edge_attrs = []
          for from_node, to_node, k, attr in nx_graph.edges(keys=True, data=True):
               node_type_from, new_id_from = node_map[from_node]
               node_type_to, new_id_to = node_map[to_node]
               edge_tp = attr['edge_type']
               weight = attr['weight']
               if edge_tp == 'word_sent':
                    if node_type_from == 'sentence':
                         sent_word_edge_indices.append([new_id_from, new_id_to])
                         sent_word_edge_attrs.append([weight])
                    else:
                         word_sent_edge_indices.append([new_id_from, new_id_to])
                         word_sent_edge_attrs.append([weight])
               elif edge_tp == 'pronoun_antecedent':
                    pro_ant_edge_indices.append([new_id_from, new_id_to])
                    pro_ant_edge_attrs.append([weight])
               elif edge_tp == 'similarity':
                    similarity_edge_indices.append([new_id_from, new_id_to])
                    similarity_edge_attrs.append([weight])
          
          het_graph['sentence', 'similarity', 'sentence'].edge_index = torch.tensor(similarity_edge_indices).t()
          het_graph['sentence', 'similarity', 'sentence'].edge_attr = torch.tensor(similarity_edge_attrs)
          het_graph['sentence', 'pro_ant', 'sentence'].edge_index = torch.tensor(pro_ant_edge_indices).t()
          het_graph['sentence', 'pro_ant', 'sentence'].edge_attr = torch.tensor(pro_ant_edge_attrs)
          het_graph['sentence', 'has', 'word'].edge_index = torch.tensor(sent_word_edge_indices).t()
          het_graph['sentence', 'has', 'word'].edge_attr = torch.tensor(sent_word_edge_attrs)
          het_graph['word', 'in', 'sentence'].edge_index = torch.tensor(word_sent_edge_indices).t()
          het_graph['word', 'in', 'sentence'].edge_attr = torch.tensor(sent_word_edge_attrs)
          
          hetro_graphs.append(het_graph)
          nodeid_to_sent_txt_map_list.append(node_map)
     
     return hetro_graphs, nodeid_to_sent_txt_map_list

def get_node_sent_map(original_graph_list, original_new_map_list):
     """_summary_ get the node in new pyg graph corresponding sentence text. to do summarization in T5.

     Args:
          original_graph_list (_type_): networkx graph list
          original_new_map_list (_type_): nx graph id to pyg hetro graph, id -> (node type, node id)
     """
     node_sent_map_list = []
     for (ori_graph, ori_new_map) in zip(original_graph_list, original_new_map_list):
          node_sent_map = {}
          for old_node in ori_graph.nodes():
               attributes = ori_graph.nodes[old_node]
               if not attributes['type'] == 'sentence': continue
               
               sent_txt = attributes['text'][2] ## only sentence text needed
               new_node_type, new_node_id = ori_new_map[old_node]
               node_sent_map[new_node_id] = sent_txt
          
          node_sent_map_list.append(node_sent_map)
          
     return node_sent_map_list