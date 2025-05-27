import torch
import networkx as nx
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from contextlib import contextmanager
import numpy as np
import multiprocessing
import torch.multiprocessing as mp
from tqdm.auto import tqdm
import time
import traceback

from models.CheckPointManager import DataCheckpointManager
from models.RelHetGraph import EdgeKeyTuple
from utils.define_node import define_node_edge_opt_parallel
from utils.model_utils import auto_workers, clean_memory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@contextmanager
def load_emb_models(models_info, target_device):
     """
     Context manager to load specified BERT models onto a device
     and ensure they are deleted from memory afterwards.
     """
     models = {}
     try:
          for model_type, model_name in models_info.items():
               if model_type == 'main_transformer':
                    models[model_type] = AutoModel.from_pretrained(model_name).to(target_device)
               elif model_type == 'tokenizer':
                    models[model_type] = AutoTokenizer.from_pretrained(model_name, use_fast=False)
               else:
                    raise ValueError(f"Unsupported model type: {model_type}")
          
          yield models

     finally:
          for model_name, model in models.items():
               del model
          
          clean_memory()
          
@contextmanager
def load_bert_models(models_info, target_device):
     """
     Context manager to load specified BERT models onto a device
     and ensure they are deleted from memory afterwards.
     """
     models = {}
     try:
          for model_type, model_name in models_info.items():
               if model_type == 'normal':
                    models[model_type] = BertModel.from_pretrained(model_name).to(target_device)
               elif model_type == 'abs_pos':
                    bert_config_abs = BertConfig.from_pretrained(model_name)
                    bert_config_abs.position_embedding_type = "absolute"
                    models[model_type] = BertModel.from_pretrained(model_name, config=bert_config_abs).to(target_device)
               elif model_type == 'rel_pos':
                    bert_config_rel = BertConfig.from_pretrained(model_name)
                    bert_config_rel.position_embedding_type = "relative_key"
                    models[model_type] = BertModel.from_pretrained(model_name, config=bert_config_rel).to(target_device)
               elif model_type == 'tokenizer':
                    models[model_type] = BertTokenizer.from_pretrained(model_name)
               elif model_type == 'sent_bert':
                    # SentenceTransformer handles device placement internally if passed at init
                    models[model_type] = SentenceTransformer(model_name, device=target_device)
               else:
                    raise ValueError(f"Unsupported model type: {model_type}")
          
          yield models

     finally:
          for model_name, model in models.items():
               del model
          
          clean_memory()

def get_sent_pos_encoding(sentid_node_map_list, bert_abs_model, bert_relative_model):
     """
     Calculates combined absolute (document) and relative (sentence)
     positional embeddings for sentence nodes. Optimized version.
     """
     sent_pos_emb_list = []
     target_device = bert_abs_model.device

     for sentid_node_map in sentid_node_map_list:
          sent_node_emb_map = {}
          if not sentid_node_map:
               sent_pos_emb_list.append(sent_node_emb_map)
               continue

          # --- Group nodes by doc_id first ---
          nodes_by_doc = {}
          max_doc_id = -1
          for key, node_id in sentid_node_map.items():
               training_id, doc_id, sent_id = key
               if doc_id not in nodes_by_doc:
                    nodes_by_doc[doc_id] = []
               nodes_by_doc[doc_id].append({'sent_id': sent_id, 'node_id': node_id})
               max_doc_id = max(max_doc_id, doc_id) # Find max doc_id efficiently

          if max_doc_id == -1: # Handle case where map had items but no valid doc_ids?
               sent_pos_emb_list.append(sent_node_emb_map)
               continue

          doc_size = 1 + max_doc_id

          # --- Calculate absolute doc embeddings ---
          with torch.no_grad():
               doc_input_ids = torch.arange(doc_size, device=target_device)
               doc_pos_embeddings = bert_abs_model.embeddings.position_embeddings(doc_input_ids)

          # --- Calculate relative embeddings ONCE per document ---
          relative_embeddings_cache = {} 
          for doc_id, nodes in nodes_by_doc.items():
               max_sent_id = max(n['sent_id'] for n in nodes)
               sent_size = max_sent_id + 1
               sent_pos_embeddings = []
               embedding_size = 512 # BERT base has 512 max position embeddings
               overlap = 5
               i = 0
               with torch.no_grad():
                    while i < sent_size:
                         start_pos = max(0, i - overlap)
                         end_pos = min(sent_size, start_pos + embedding_size)
                         batch_indices = torch.arange(end_pos - start_pos, device=target_device)

                         batch_emb = bert_relative_model.embeddings.position_embeddings(batch_indices)

                         actual_overlap = min(overlap, batch_emb.size(0)) 
                         next_id_idx = 0 if start_pos == 0 else actual_overlap

                         sent_pos_embeddings.extend(batch_emb[next_id_idx:])
                         i = end_pos
               # Trim and store in cache
               relative_embeddings_cache[doc_id] = torch.stack(sent_pos_embeddings[:sent_size])

               # --- Assign combined embeddings for nodes in this doc ---
               doc_abs_emb = doc_pos_embeddings[doc_id]
               doc_rel_embs = relative_embeddings_cache[doc_id]
               for node_info in nodes:
                    sent_id = node_info['sent_id']
                    node_id = node_info['node_id']

                    node_embedding = doc_abs_emb + doc_rel_embs[sent_id]
                    sent_node_emb_map[node_id] = node_embedding # Keep on model's device

          sent_pos_emb_list.append(sent_node_emb_map)

     return sent_pos_emb_list

def parallel_create_graph(args):
     """
     Worker function for multiprocessing: Creates a single NetworkX graph.
     Takes a tuple of (word_node_map, sent_node_map, edges_data) as input.
     Ensures graph attributes are picklable.
     """
     try:
          word_node_map, sent_node_map, edges_data, doc_sents_map = args
          graph = nx.MultiDiGraph()

          # Add word nodes
          word_nodes = [(w_node_id, {"type": "word", "text": word})
                         for word, w_node_id in word_node_map.items()]
          graph.add_nodes_from(word_nodes)

          # Add sentence nodes
          sent_nodes = [(s_node_id, {"type": "sentence", "text": sent_triple})
                         for sent_triple, s_node_id_list in sent_node_map.items()
                         for s_node_id in s_node_id_list]
          graph.add_nodes_from(sent_nodes)

          # Add document nodes
          doc_nodes = [(doc, {"type": "document", "sents": sents}) for doc, sents in doc_sents_map.items()]
          graph.add_nodes_from(doc_nodes)
          
          # Add edges
          edges = []
          for (node1, node2), edge_list in edges_data.items():
               for edge in edge_list:
                    edge_attr = {
                         "edge_type": edge["type"],
                         "weight": edge["weight"]
                    }
                    edges.append((node1, node2, edge_attr))
                    edges.append((node2, node1, edge_attr))

          graph.add_edges_from(edges)
          
          return graph
     
     except Exception as e:
          print(f"[ERROR] An exception occurred while creating graphs: {e}")
          traceback.print_exc() 

def embed_nodes_with_rel_pos(graphs, word_emb_batch_size=64, sentence_emb_batch_size=32):
     """embed nodes with explicit no positional encoding
     """
     models_info = {
          'main_transformer': 'microsoft/deberta-v3-base',
          'tokenizer': 'microsoft/deberta-v3-base',
     }
     embedded_graphs = []
     
     print(f"Embedding nodes on device: {device}")
     
     try:
          with load_emb_models(models_info, device) as models:
               main_transformer_model = models["main_transformer"].eval()
               tokenizer = models["tokenizer"]
               
               try:
                    target_dim = main_transformer_model.config.hidden_size
                    if not isinstance(target_dim, int) or target_dim <= 0:
                         raise ValueError("Invalid hidden_size")
               except AttributeError:
                    raise ValueError("Could not determine hidden_size from main_transformer_model config.")

               for i, graph in enumerate(tqdm(graphs, desc="Embedding Graphs", position=0, leave=True)):
                    sentences_texts = []
                    sentence_node_ids = []
                    word_texts = []
                    word_node_ids = []
                    doc_node_ids = []
                    doc_sents_ids = []

                    for node, data in graph.nodes(data=True):
                         node_type = data.get('type')
                         node_text_data = data.get('text')

                         if node_type == 'sentence':
                              sentences_texts.append(node_text_data[2])
                              sentence_node_ids.append(node)
                         elif node_type == 'word':
                              word_texts.append(node_text_data)
                              word_node_ids.append(node)
                         elif node_type == 'document':
                              doc_node_ids.append(node)
                              doc_sents_ids.append(data.get('sents'))
                         
                    
                    # --- Sentence Embedding ---
                    final_sentence_embed_id_map = {}
                    if sentence_node_ids:
                         all_sent_embeddings = []
                         with torch.no_grad():
                              for j in range(0, len(sentences_texts), sentence_emb_batch_size):
                                   batch_texts = sentences_texts[j : j + sentence_emb_batch_size]

                                   inputs = tokenizer(
                                        batch_texts,
                                        return_tensors='pt',
                                        padding=True,
                                        truncation=True,
                                        max_length=main_transformer_model.config.max_position_embeddings # Use model's capacity
                                   ).to(device)

                                   outputs = main_transformer_model(**inputs)
                                   token_embeddings = outputs.last_hidden_state # shape: (batch, seq_len, hidden_size)

                                   # mean pool token embeddings (respecting padding) to get sentence embedding
                                   input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                                   sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                                   sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                                   mean_pooled_embeddings = sum_embeddings / sum_mask # Shape: (batch, hidden_size)
                                   all_sent_embeddings.append(mean_pooled_embeddings)

                         if all_sent_embeddings:
                              final_main_sent_embeddings = torch.cat(all_sent_embeddings, dim=0)
                         else:
                              print("[Warning] Sentence embedding EMPTY!")
                         
                         # Map embeddings directly to node IDs
                         for idx, node_id in enumerate(sentence_node_ids):
                              final_sentence_embed_id_map[node_id] = final_main_sent_embeddings[idx]
                    else:
                         print(f"[Warning] No sentence embeddings generated for graph {i}.")


                    # --- Doc Embedding --
                    final_doc_embed_id_map = {}
                    if sentence_node_ids:
                         ## average sent_emb to get doc_emb
                         for idx, doc_id in enumerate(doc_node_ids):
                              temp_sent_embs = []
                              for sent_id in doc_sents_ids[idx]:
                                   temp_sent_embs.append(final_sentence_embed_id_map[sent_id])
                              
                              if temp_sent_embs:
                                   doc_embedding = torch.mean(torch.stack(temp_sent_embs, dim=0), dim=0)
                              else:
                                   sent_node, sent_emb = next(iter(final_sentence_embed_id_map.items()))
                                   doc_embedding = torch.zeros_like(sent_emb)

                              final_doc_embed_id_map[doc_id] = doc_embedding
                    else:
                         print(f"[Warning] No document embeddings generated for graph {i}.")

                    
                    # --- Word Embedding ---
                    final_word_embed_id_map = {}
                    if word_node_ids:
                         all_word_embeddings = []
                         with torch.no_grad():
                              for j in range(0, len(word_texts), word_emb_batch_size):
                                   batch_texts = word_texts[j : j + word_emb_batch_size]

                                   inputs = tokenizer(
                                        batch_texts, return_tensors='pt', padding=True, truncation=True,
                                        max_length=main_transformer_model.config.max_position_embeddings
                                   ).to(device)

                                   outputs = main_transformer_model(**inputs)
                                   token_embeddings = outputs.last_hidden_state
                                   # Mean Pooling
                                   input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                                   sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                                   sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                                   mean_pooled_embeddings = sum_embeddings / sum_mask
                                   all_word_embeddings.append(mean_pooled_embeddings)

                         if all_word_embeddings:
                              final_word_embeddings = torch.cat(all_word_embeddings, dim=0)

                         for idx, node_id in enumerate(word_node_ids):
                              final_word_embed_id_map[node_id] = final_word_embeddings[idx]
                    
                         del all_word_embeddings
                    else:
                         print(f"\nWarning: No word embeddings generated for graph {i} (List was empty).")

                    # --- Assign Embeddings to Graph Nodes ---
                    for node_id, node_data in graph.nodes(data=True):
                         if node_data.get('type') == 'sentence':
                              if node_id in final_sentence_embed_id_map:
                                   graph.nodes[node_id]['embedding'] = final_sentence_embed_id_map[node_id]
                         elif node_data.get('type') == 'word':
                              if node_id in final_word_embed_id_map:
                                   graph.nodes[node_id]['embedding'] = final_word_embed_id_map[node_id]
                         elif node_data.get('type') == 'document':
                              if node_id in final_doc_embed_id_map:
                                   graph.nodes[node_id]['embedding'] = final_doc_embed_id_map[node_id]

                    embedded_graphs.append(graph)
                    
                    del final_word_embed_id_map, final_sentence_embed_id_map
     
     except Exception as e:
          print(f"[ERROR] An exception occurred while embedding graphs: {e}")
          traceback.print_exc()
     
     finally:
          return embedded_graphs
     
     
def embed_nodes_with_abs_pos(graphs, sentid_node_map_list, word_batch_size=64, sentence_batch_size=32):
     """
     Embeds nodes using BERT models on GPU.
     - Batches word embeddings for efficiency.
     - Assigns sentence embeddings efficiently.
     - Returns graphs with embeddings attached (still on GPU).
     """
     models_info = {
          'normal': 'bert-base-uncased',
          'abs_pos': 'bert-base-uncased',
          'rel_pos': 'bert-base-uncased',
          'tokenizer': 'bert-base-uncased',
          'sent_bert': 'all-MiniLM-L6-v2',
     }
     embedded_graphs = []
     
     print(f"Embedding nodes on device: {device}")

     with load_bert_models(models_info, device) as models:
          # Get models from context manager
          bert_abs_model = models["abs_pos"].eval()
          bert_relative_model = models["rel_pos"].eval()
          bert_tokenizer = models["tokenizer"]
          bert_model = models["normal"].eval()
          sentBERT_model = models["sent_bert"] # SentenceTransformer handles eval mode internally

          # Pre-calculate all positional embeddings first
          sent_node_embedding_map_list = get_sent_pos_encoding(sentid_node_map_list, bert_abs_model, bert_relative_model)

          # Process each graph
          for i, graph in enumerate(graphs):
               sent_node_embedding_map = sent_node_embedding_map_list[i]

               # --- Sentence Embedding (Batched) ---
               sentences_data = [] # Store (node_id, text, position_embedding)
               sentence_nodes_to_embed = [] # Store node ids that need SBERT embedding
               word_texts = []
               word_node_ids = []
               for node, data in graph.nodes(data=True):
                    if data['type'] == 'sentence':
                         if node in sent_node_embedding_map:
                              pos_embedding = sent_node_embedding_map[node] # Get pre-calculated pos embedding
                              sentences_data.append((node, data['text'][2], pos_embedding)) # Use text part for SBERT
                              sentence_nodes_to_embed.append(node)
                         else:
                              print(f"Warning: Node {node} (sentence) not found in positional embedding map for graph {i}. Skipping.")
                    elif data['type'] == 'word':
                         word_texts.append(data['text'])
                         word_node_ids.append(node)

               if sentences_data:
                    sentence_texts = [text for _, text, _ in sentences_data]
                    with torch.no_grad():
                         sent_embeddings = sentBERT_model.encode(
                         sentence_texts,
                         convert_to_tensor=True,
                         normalize_embeddings=True,
                         batch_size=sentence_batch_size,
                         show_progress_bar=False # Reduce console noise
                         )

                    # Combine SBERT embedding (384) + Positional (768)
                    # Pad SBERT to match positional embedding dimension (768)
                    sbert_dim = sent_embeddings.shape[1]
                    target_dim = bert_abs_model.config.hidden_size # Should be 768 for base
                    
                    if sbert_dim < target_dim:
                         padding_size = target_dim - sbert_dim
                         padding = torch.zeros(sent_embeddings.shape[0], padding_size, device=sent_embeddings.device)
                         padded_sent_emb = torch.cat([sent_embeddings, padding], dim=1)
                    elif sbert_dim == target_dim:
                         padded_sent_emb = sent_embeddings
                    else:
                         # This case shouldn't happen with MiniLM (384) and BERT base (768)
                         # If models change, may need adjustment (e.g., truncation or projection)
                         print(f"Warning: SBERT dim ({sbert_dim}) > Target dim ({target_dim}). Using SBERT embedding directly.")
                         padded_sent_emb = sent_embeddings

                    # Assign combined embeddings
                    for idx, (node_id, _, pos_embedding) in enumerate(sentences_data):
                         graph.nodes[node_id]['embedding'] = pos_embedding.to(padded_sent_emb.device) + padded_sent_emb[idx]

               # --- Word Embedding (Batched) ---
               if word_node_ids:
                    all_word_embeddings = [] # Store embeddings in order

                    with torch.no_grad():
                         for j in range(0, len(word_texts), word_batch_size):
                              batch_texts = word_texts[j : j + word_batch_size]
                              
                              # Tokenize batch
                              word_tokens = bert_tokenizer(
                                   batch_texts,
                                   return_tensors='pt',
                                   padding=True,       # Pad sequences to max length in batch
                                   truncation=True,    # Truncate if longer than model max length
                                   max_length=bert_model.config.max_position_embeddings # Use model's max length
                              ).to(device)

                              # Get token embeddings from BERT
                              token_embeddings = bert_model(**word_tokens).last_hidden_state # (batch_size, num_tokens, 768)

                              # Calculate mean embedding, masking padding tokens
                              input_mask_expanded = word_tokens['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                              sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                              # Clamp sum_mask to avoid division by zero for empty sequences (shouldn't happen with tokenizer)
                              sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                              mean_batch_embeddings = sum_embeddings / sum_mask # (batch_size, 768)
                              all_word_embeddings.append(mean_batch_embeddings)

                    # Concatenate embeddings from all batches
                    if all_word_embeddings:
                         final_word_embeddings = torch.cat(all_word_embeddings, dim=0)
                         for idx, node_id in enumerate(word_node_ids):
                              graph.nodes[node_id]['embedding'] = final_word_embeddings[idx] # Keep on GPU

               embedded_graphs.append(graph)

     return embedded_graphs

def parallel_convert_graph_serializable(nx_graph):
     """
     Worker function for multiprocessing: Converts a single NetworkX graph
     (with CPU embeddings) to a PyTorch Geometric HeteroData object (on CPU).
     Also generates necessary mappings.
     """

     node_map = {}
     sent_id = 0
     word_id = 0
     doc_id = 0
     word_nodes_feat = []
     word_texts = []
     sent_nodes_feat = []
     sent_texts = []
     doc_nodes_feat = []
     
     try:
          for node, data in nx_graph.nodes(data=True):
               cur_type = data['type']
               embed = data.get('embedding', None)
               if embed is None:
                    print(f"Warning: Node {node} of type {cur_type} has no 'embedding' attribute.")
                    pass
                         
               if cur_type == 'word':
                    if embed is not None:
                         word_nodes_feat.append(embed)
                    word_texts.append(str(data['text']))
                    node_map[node] = ('word', word_id)
                    word_id += 1
               elif cur_type == 'sentence':
                    if embed is not None:
                         sent_nodes_feat.append(embed)
                    sent_texts.append(str(data['text'][2]))
                    node_map[node] = ('sentence', sent_id)
                    sent_id += 1
               elif cur_type == 'document':
                    if embed is not None:
                         doc_nodes_feat.append(embed)
                    node_map[node] = ('document', doc_id)
                    doc_id += 1

          serializable_graph_data = {
               'node_types': {},
               'edge_types': {}
          }

          if word_id > 0:
               serializable_graph_data['node_types']['word'] = {'text': word_texts}
               if word_nodes_feat:
                    serializable_graph_data['node_types']['word']['x'] = np.stack(word_nodes_feat)

          if sent_id > 0:
               serializable_graph_data['node_types']['sentence'] = {'text': sent_texts}
               if sent_nodes_feat:
                    serializable_graph_data['node_types']['sentence']['x'] = np.stack(sent_nodes_feat)
          
          if doc_id > 0:
               serializable_graph_data['node_types']['document'] = {}
               if doc_nodes_feat:
                    serializable_graph_data['node_types']['document']['x'] = np.stack(doc_nodes_feat)
          
               
          edge_data_temp = {
               EdgeKeyTuple.SENT_SIM.value: {'indices': [], 'attrs': []},
               EdgeKeyTuple.SENT_ANT.value: {'indices': [], 'attrs': []},
               EdgeKeyTuple.SENT_WORD.value: {'indices': [], 'attrs': []},
               EdgeKeyTuple.WORD_SENT.value: {'indices': [], 'attrs': []},
               EdgeKeyTuple.DOC_SENT.value: {'indices': [], 'attrs': []},
               EdgeKeyTuple.SENT_DOC.value: {'indices': [], 'attrs': []},
          }

          for from_node, to_node, k, attr in nx_graph.edges(keys=True, data=True):
               if from_node not in node_map or to_node not in node_map:
                    print(f"Warning: Skipping edge ({from_node}, {to_node}) due to missing node in node_map.")
                    continue

               node_type_from, new_id_from = node_map[from_node]
               node_type_to, new_id_to = node_map[to_node]
               edge_tp_attr = attr.get('edge_type')
               weight = attr.get('weight', 1.0)

               current_edge_key = None
               if edge_tp_attr == 'word_sent':
                    if node_type_from == 'sentence':
                         current_edge_key = EdgeKeyTuple.SENT_WORD.value
                    else:
                         current_edge_key = EdgeKeyTuple.WORD_SENT.value
               elif edge_tp_attr == 'pronoun_antecedent':
                    current_edge_key = EdgeKeyTuple.SENT_ANT.value
               elif edge_tp_attr == 'similarity':
                    current_edge_key = EdgeKeyTuple.SENT_SIM.value
               elif edge_tp_attr == 'doc_sent':
                    if node_type_from == 'document':
                         current_edge_key = EdgeKeyTuple.DOC_SENT.value
                    else:
                         current_edge_key = EdgeKeyTuple.SENT_DOC.value

               if current_edge_key and current_edge_key in edge_data_temp:
                    edge_data_temp[current_edge_key]['indices'].append([new_id_from, new_id_to])
                    edge_data_temp[current_edge_key]['attrs'].append([weight])
               else:
                    print(f"Warning: Unhandled or unknown edge type '{edge_tp_attr}' or key '{current_edge_key}'")


          def _build_edge_index_numpy(edge_list):
               if not edge_list:
                    return np.empty((2, 0), dtype=np.int64)
               return np.array(edge_list, dtype=np.int64).T

          for edge_key_tuple, data in edge_data_temp.items():
               if data['indices']:
                    serializable_graph_data['edge_types'][edge_key_tuple] = {
                         'edge_index': _build_edge_index_numpy(data['indices']),
                         'edge_attr': np.array(data['attrs'], dtype=np.float32)
                    }

          # Mapping from new pyg sentence id -> sentence text
          pyg_id_to_sent_txt_map = {}
          for old_node, attributes in nx_graph.nodes(data=True):
               if not attributes['type'] == 'sentence': continue
               
               sent_txt = attributes['text'][2] ## only sentence text needed
               new_node_type, new_node_id = node_map[old_node]
               pyg_id_to_sent_txt_map[new_node_id] = sent_txt
          
          
          del word_nodes_feat, sent_nodes_feat, word_texts, sent_texts, edge_data_temp, node_map

          return serializable_graph_data, pyg_id_to_sent_txt_map
     
     except Exception as e:
          print(f"[ERROR] An exception occurred while converting graphs: {e}")
          traceback.print_exc()
          return e

def get_embedded_pyg_graphs(dataset_type, docs_list, sent_similarity):
     data_cpt = DataCheckpointManager()
     num_workers = auto_workers()
     TASK_PREFIX = '[Create Graph]'
     
     print(f"Using {num_workers} workers to process {len(docs_list)} samples.")
     
     try:
          mp.set_sharing_strategy('file_system')
     except RuntimeError:
          print("Could not set PyTorch multiprocessing sharing strategy.")
     
     if (latest_step := data_cpt.get_latest_step(dataset_type = dataset_type)):
          print(f"{TASK_PREFIX} Resume from step: [{latest_step}] for {dataset_type} dataset")
     else:
          print(f"{TASK_PREFIX} Starting from scratch for {dataset_type} dataset.")

     define_node_key = data_cpt.StepKey.PREDEFINE.value
     graph_create_key = data_cpt.StepKey.GRAPH.value
     embed_graph_key = data_cpt.StepKey.EMBED.value
     final_graph_key = data_cpt.StepKey.FINAL.value

     # --- Step 1: Define Nodes and Edges ---
     word_nodeId_list, sent_nodeId_list, edge_data_list, sentid_node_map_list = None, None, None, None

     if not latest_step or latest_step in [define_node_key]:
          if not latest_step:
               print(f"{TASK_PREFIX} Step 1: Defining nodes and edges for {dataset_type} dataset...")
               start_time = time.time()

               word_nodeId_list, sent_nodeId_list, edge_data_list, sentid_node_map_list, doc_sents_map_list = define_node_edge_opt_parallel(docs_list, sent_similarity)

               data_cpt.save_step(step_name=define_node_key, data={
                    'word_nodeId_list': word_nodeId_list,
                    'sent_nodeId_list': sent_nodeId_list,
                    'edge_data_list': edge_data_list,
                    'sentid_node_map_list': sentid_node_map_list,
                    'doc_sents_map_list': doc_sents_map_list
               }, dataset_type=dataset_type)
               print(f"{TASK_PREFIX} Step 1 finished in {time.time() - start_time:.4f}s.")
          else:
               data = data_cpt.load_step(step_name=define_node_key, dataset_type=dataset_type)
               if data:
                    word_nodeId_list = data['word_nodeId_list']
                    sent_nodeId_list = data['sent_nodeId_list']
                    edge_data_list = data['edge_data_list']
                    sentid_node_map_list = data['sentid_node_map_list']
                    doc_sents_map_list = data['doc_sents_map_list']

     ############333test
     print(f"after Step 1: word_nodeId_list: {len(word_nodeId_list)}, sent_nodeId_list: {len(sent_nodeId_list)}, edge_data_list: {len(edge_data_list)}, sentid_node_map_list: {len(sentid_node_map_list)}, doc_sents_map_list: {len(doc_sents_map_list)}")
     ########################
     # --- Step 2: Create NetworkX Graphs (Parallel) ---
     graph_list = None
     if not latest_step or latest_step in [define_node_key, graph_create_key]:
          if not latest_step or latest_step != graph_create_key:  # Run if starting or finished step 1
               print(f"{TASK_PREFIX} Step 2: Creating NetworkX graphs in parallel for {dataset_type} dataset...")
               start_time = time.time()
               num_items = len(word_nodeId_list)
               pool_args = list(zip(word_nodeId_list, sent_nodeId_list, edge_data_list, doc_sents_map_list))

               graph_list = []
               if num_items > 0:
                    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
                         for nx_graph in tqdm(pool.imap(parallel_create_graph, pool_args), total=num_items, desc="Creating NX Graphs"):
                              graph_list.append(nx_graph)

               data_cpt.save_step(step_name=graph_create_key, data={'graph_list': graph_list}, dataset_type=dataset_type)
               print(f"{TASK_PREFIX} Step 2 finished in {time.time() - start_time:.4f}s.")
          else:
               data = data_cpt.load_step(step_name=graph_create_key,dataset_type=dataset_type)
               if data:
                    graph_list = data['graph_list']
                    define_data = data_cpt.load_step(step_name=define_node_key, dataset_type=dataset_type)
                    if define_data:
                         sentid_node_map_list = define_data['sentid_node_map_list']
     
     ############333test
     print(f"after Step 2: graph_list: {len(graph_list)}")
     ########################
     
     # --- Step 3: Embed Graph Nodes (Batched, on GPU) ---
     embedded_graph_list = None
     if not latest_step or latest_step in [define_node_key, graph_create_key, embed_graph_key]:
          if not latest_step or latest_step != embed_graph_key:  # Run if starting or finished step 1 or 2
               print(f"{TASK_PREFIX} Step 3: Embedding graph nodes for {dataset_type} dataset...")
               start_time = time.time()
               embedded_graphs_gpu = embed_nodes_with_rel_pos(graph_list)

               ## transfers to cpu
               embedded_graph_list = []
               for g in embedded_graphs_gpu:
                    for node, data in g.nodes(data=True):
                         if 'embedding' in data and isinstance(data['embedding'], torch.Tensor):
                              data['embedding'] = data['embedding'].detach().cpu()
                    embedded_graph_list.append(g)

               del embedded_graphs_gpu

               data_cpt.save_step(step_name=embed_graph_key, data={'embedded_graph_list': embedded_graph_list}, dataset_type=dataset_type)
               print(f"{TASK_PREFIX} Step 3 finished in {time.time() - start_time:.4f}s.")
          else:
               data = data_cpt.load_step(step_name=embed_graph_key, dataset_type=dataset_type)
               if data:
                    embedded_graph_list = data['embedded_graph_list']
                    for g in embedded_graph_list:
                         for node, data in g.nodes(data=True):
                              if 'embedding' in data and isinstance(data['embedding'], torch.Tensor) and data['embedding'].requires_grad:
                                   data['embedding'] = data['embedding'].detach()

     ############333test
     print(f"after Step 3: embedded_graph_list: {len(embedded_graph_list)}")
     ########################
     
     # --- Step 4: Convert to PyG HeteroData (Parallel) ---
     node_sent_map_list = None
     pyg_graph_list_cpu = None
     if not latest_step or latest_step in [define_node_key, graph_create_key, embed_graph_key, final_graph_key]:
          if not latest_step or latest_step != final_graph_key:
               print(f"{TASK_PREFIX} Step 4: Converting graphs to PyG format for {dataset_type} dataset...")
               start_time = time.time()
               pyg_graph_list_cpu = []
               node_sent_map_list = []
               num_items = len(embedded_graph_list)
               
               ## serialize the graph data to avoid memory issues
               for g in embedded_graph_list:
                    for node, data in g.nodes(data=True):
                         if 'embedding' in data and isinstance(data['embedding'], torch.Tensor):
                              data['embedding'] = data['embedding'].detach().cpu().numpy()
               
               if num_items > 0:
                    with multiprocessing.get_context("spawn").Pool(num_workers) as pool, \
                         tqdm(total=num_items, desc="Converting to PyG") as pbar:
                    
                         for res_idx, res in enumerate(pool.imap(parallel_convert_graph_serializable, embedded_graph_list)):
                              if isinstance(res, Exception):
                                   print(f"[Error] {TASK_PREFIX} Fail to convert {res_idx}-th graph.")
                                   pyg_graph_list_cpu.append(None)
                                   node_sent_map_list.append(None)
                              else:
                                   serializable_graph_data, pyg_id_to_sent_txt_map = res
                                   
                                   ## convert to hetero data
                                   het_graph = HeteroData()

                                   # Process Node
                                   for node_type, node_data in serializable_graph_data['node_types'].items():
                                        if 'text' in node_data:
                                             het_graph[node_type].text = node_data['text']
                                        if 'x' in node_data:
                                             het_graph[node_type].x = torch.from_numpy(node_data['x']).detach()
                                   
                                   # Process Edge
                                   for edge_key_tuple, edge_data in serializable_graph_data['edge_types'].items():
                                        het_graph[edge_key_tuple].edge_index = torch.from_numpy(edge_data['edge_index']).detach()
                                        het_graph[edge_key_tuple].edge_attr = torch.from_numpy(edge_data['edge_attr']).detach()

                                   pyg_graph_list_cpu.append(het_graph)
                                   node_sent_map_list.append(pyg_id_to_sent_txt_map)
                              
                              pbar.update()

               else:
                    pyg_graph_list_cpu = []
                    node_sent_map_list = []
                    
               data_cpt.save_step(step_name=final_graph_key, data={
                    'pyg_graph_list': pyg_graph_list_cpu,
                    'node_sent_map_list': node_sent_map_list},
                    dataset_type=dataset_type)
               print(f"{TASK_PREFIX} Step 4 finished in {time.time() - start_time:.4f}s.")
          else:
               data = data_cpt.load_step(step_name=final_graph_key, dataset_type=dataset_type)
               pyg_graph_list_cpu = data['pyg_graph_list']
               node_sent_map_list = data['node_sent_map_list']
     
     ############333test
     print(f"after Step 4: pyg_graph_list_cpu: {len(pyg_graph_list_cpu)}, node_sent_map_list: {len(node_sent_map_list)}")
     ########################
     
     clean_memory()
     
     return pyg_graph_list_cpu, node_sent_map_list