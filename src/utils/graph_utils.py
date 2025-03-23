import networkx as nx
import torch
import time
from transformers import BertTokenizer, BertModel, BertConfig
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from contextlib import contextmanager

from utils.data_preprocess_utils import define_node_edge, load_jsonl
from utils.model_utils import clean_memory, print_gpu_memory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_embed_graph(file_path):
     docs_list, summary_list = load_jsonl(file_path)
     print("[preprocess] Data file is loaded. Creating embedding graph...")
     start = time.time()
     sample_graphs, node_maps = create_embed_graphs(docs_list)
     end = time.time()
     print(f"[preprocess] Finish graph creation, time cost:  {end - start:.4f} s.")
     
     clean_memory()
     print_gpu_memory("after graph embedding")
     return sample_graphs

def get_embed_graph_node_map(file_path):
     docs_list, summary_list = load_jsonl(file_path)
     print(f"Data file is loaded. Creating embedding graph and node mapping...")
     start = time.time()
     sample_graphs, node_maps = create_embed_graphs(docs_list)
     end = time.time()
     print(f"Finish graph creation, time cost:  {end - start:.4f} s.")
     
     clean_memory()
     print_gpu_memory("after graph embedding")
     return sample_graphs, node_maps, summary_list

def create_embed_graphs(docs_list, sent_similarity = 0.6):
     word_nodeId_list, sent_nodeId_list, edge_data_list, sentid_node_map_list = define_node_edge(docs_list, sent_similarity)
     graph_list = create_graph(word_nodeId_list, sent_nodeId_list, edge_data_list)
     embedded_graph_list = embed_nodes_gpu(graph_list, sentid_node_map_list)
     pyg_graph_list, nodeid_to_sent_map_list = convert_graph_from_nx_to_pyg(embedded_graph_list)
     node_sent_map_list = get_node_sent_map(embedded_graph_list, nodeid_to_sent_map_list) ## id -> sent
     
     return pyg_graph_list, node_sent_map_list

def create_graph(word_nodeId_list, sent_nodeId_list, edge_data_list):
     graph_list = []
     
     for (word_node_map, sent_node_map, edges_data) in zip(word_nodeId_list, sent_nodeId_list, edge_data_list):
          ## create graph for each multi-doc training sample
          graph = nx.MultiDiGraph()
          
          word_nodes = [(w_node_id, {"type": "word", "text": word}) for word, w_node_id in word_node_map.items()]
          graph.add_nodes_from(word_nodes)

          sent_nodes = [(s_node_id, {"type": "sentence", "text": sent_triple}) for sent_triple, s_node_id_list in sent_node_map.items() for s_node_id in s_node_id_list]
          graph.add_nodes_from(sent_nodes)
          
          edges = []
          for (node1, node2), edge_list in edges_data.items():
               for edge in edge_list:
                    edges.append((node1, node2, {"edge_type": edge["type"], "weight": edge["weight"]}))
                    edges.append((node2, node1, {"edge_type": edge["type"], "weight": edge["weight"]}))

          graph.add_edges_from(edges)
          
          graph_list.append(graph)
     
     return graph_list

@contextmanager
def load_bert_models(models_info, device):
     models = {}

     try:
          for model_type, model_name in models_info.items():
               if model_type == 'normal':
                    models[model_type] = BertModel.from_pretrained(model_name).to(device)
               elif model_type == 'abs_pos':
                    bert_config_abs = BertConfig.from_pretrained(model_name)
                    bert_config_abs.position_embedding_type = "absolute"
                    models[model_type] = BertModel(bert_config_abs).to(device)
               elif model_type == 'rel_pos':
                    bert_config_rel = BertConfig.from_pretrained(model_name)
                    bert_config_rel.position_embedding_type = "relative_key"
                    models[model_type] = BertModel(bert_config_rel).to(device)
               elif model_type == 'tokenizer':
                    models[model_type] = BertTokenizer.from_pretrained(model_name)
               elif model_type == 'sent_bert':
                    models[model_type] = SentenceTransformer(model_name, device=device)
               else:
                    raise ValueError(f"Unsupported model type: {model_type}")
          
          yield models

     finally:
          # clean up models after use
          for model in models.values():
               del model
          

def embed_nodes_gpu(graphs, sentid_node_map_list):
     """Embeds nodes in the graph using SBERT, BERT, and positional embeddings."""
     models_info = {
          'normal': 'bert-base-uncased',
          'abs_pos': 'bert-base-uncased',
          'rel_pos': 'bert-base-uncased',
          'tokenizer': 'bert-base-uncased',
          'sent_bert': 'all-MiniLM-L6-v2',
     }
     embedded_graphs = []

     with load_bert_models(models_info, device) as models: ## model gpu memory control
          bert_abs_model = models["abs_pos"]
          bert_relative_model = models["rel_pos"]
          bert_tokenizer = models["tokenizer"]
          bert_model = models["normal"]
          sentBERT_model = models["sent_bert"]
          ## eval mode
          bert_abs_model.eval()
          bert_relative_model.eval()
          bert_model.eval()
          sentBERT_model.eval()
          
          sent_node_embedding_map_list = get_sent_pos_encoding(sentid_node_map_list, bert_abs_model, bert_relative_model)
          for graph, sent_node_embedding_map in zip(graphs, sent_node_embedding_map_list):
               sentences = [data['text'][2] for node, data in graph.nodes(data=True) if data['type'] == 'sentence']
               if sentences:
                    with torch.no_grad():
                         # encode all sentences in a batch
                         sent_embeddings = sentBERT_model.encode(
                              sentences, 
                              convert_to_tensor=True, 
                              normalize_embeddings=True, 
                              batch_size=32,
                              device=device)  # (num_sentences, 384)

               sent_idx = 0
               for node, data in graph.nodes(data=True):
                    if data['type'] == 'sentence':
                         with torch.no_grad():
                              position_embeddings = sent_node_embedding_map[node]  # (768)
                              sent_embedding = sent_embeddings[sent_idx]  # (384,)
                              sent_idx += 1

                              # Combine embeddings
                              pad_sent_emb = torch.cat([sent_embedding, torch.zeros(384, device=device)], dim=0)  # (768,)
                              graph.nodes[node]['embedding'] = position_embeddings + pad_sent_emb

                    elif data['type'] == 'word':
                         with torch.no_grad():
                              word_tokens = bert_tokenizer(graph.nodes[node]['text'], return_tensors='pt').to(device)
                              token_embeddings = bert_model(**word_tokens).last_hidden_state  # (1, num_tokens, 768)
                              word_embedding = torch.mean(token_embeddings, dim=1).squeeze()  # (768,)

                              graph.nodes[node]['embedding'] = word_embedding

               embedded_graphs.append(graph)

          del sentBERT_model
     
          return embedded_graphs

def get_sent_pos_encoding(sentid_node_map_list, bert_abs_model, bert_relative_model):
     sent_pos_emb_list = []
     
     for sentid_node_map in sentid_node_map_list:
          sent_node_emb_map = {}
          doc_sentinel = -1
          doc_size = 1 + next(reversed(sentid_node_map.items()))[0][1] ## the second key is doc_id
          # doc absolute position embedding
          doc_input_ids = [i for i in range(doc_size)]
          doc_pos_embeddings = bert_abs_model.embeddings.position_embeddings(torch.tensor(doc_input_ids).to(device))
          sent_pos_embeddings = []

          for (training_id, doc_id, sent_id), node_id in reversed(sentid_node_map.items()):
               if(doc_id == doc_sentinel):
                    node_embedding = doc_pos_embeddings[doc_id] + sent_pos_embeddings[sent_id]
                    sent_node_emb_map[node_id] = node_embedding
                    continue
               
               doc_sentinel = doc_id

               # sentence relative position embedding
               # sent_input_ids = [i for i in range(sent_id + 1)]
               sent_pos_embeddings = []
               embedding_size = 512
               sent_size = sent_id + 1
               overlap = 5
               i = 0
               while i < sent_size: ## in case the sent number exceeds the embedding size:512
                    start_pos = max(0, i - overlap)
                    end_pos = min(sent_size, start_pos + embedding_size)
                    # batch = sent_input_ids[start_pos:end_pos]
                    batch = [j for j in range((end_pos - start_pos))]
                    
                    batch_emb = bert_relative_model.embeddings.position_embeddings(torch.tensor(batch).to(device))
                    next_id = 0 if start_pos == 0 else overlap
                    sent_pos_embeddings.extend(batch_emb[next_id:]) ## simplely cut
                    i = end_pos
                    
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
          word_texts = []
          sent_nodes = []
          sent_texts = []
          for node in nx_graph.nodes():
               cur_type = nx_graph.nodes[node]['type']
               embed = nx_graph.nodes[node]['embedding']
               text = nx_graph.nodes[node]['text']
               if cur_type == 'word':
                    word_nodes.append(embed)
                    word_texts.append(text) # word
                    node_map[node] = ('word', word_id)
                    word_id = word_id + 1
               elif cur_type == 'sentence':
                    sent_nodes.append(embed)
                    sent_texts.append(text[2])# (training id, doc id, text)
                    node_map[node] = ('sentence', sent_id)
                    sent_id = sent_id + 1
          het_graph['word'].x = torch.stack(word_nodes)
          het_graph['word'].text = word_texts
          het_graph['sentence'].x = torch.stack(sent_nodes)
          het_graph['sentence'].text = sent_texts
          
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
          
          het_graph['sentence', 'similarity', 'sentence'].edge_index = torch.tensor(similarity_edge_indices).t().to(torch.int64)
          het_graph['sentence', 'similarity', 'sentence'].edge_attr = torch.tensor(similarity_edge_attrs)
          het_graph['sentence', 'pro_ant', 'sentence'].edge_index = torch.tensor(pro_ant_edge_indices).t().to(torch.int64)
          het_graph['sentence', 'pro_ant', 'sentence'].edge_attr = torch.tensor(pro_ant_edge_attrs)
          het_graph['sentence', 'has', 'word'].edge_index = torch.tensor(sent_word_edge_indices).t().to(torch.int64)
          het_graph['sentence', 'has', 'word'].edge_attr = torch.tensor(sent_word_edge_attrs)
          het_graph['word', 'in', 'sentence'].edge_index = torch.tensor(word_sent_edge_indices).t().to(torch.int64)
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