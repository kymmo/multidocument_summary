import torch
import concurrent.futures
from itertools import zip_longest
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import time
import traceback
import threading
import spacy
import psutil
import multiprocessing

from utils.data_preprocess_utils import generator_extract_keywords
from utils.model_utils import clean_memory, print_cpu_memory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlp_model_name = "en_core_web_lg"
sent_model_name = 'all-MiniLM-L6-v2'

## main process model define
_main_coref_nlp = None
_main_st_model = None
def get_main_coref_nlp():
     global _main_coref_nlp
     if _main_coref_nlp is None:
          _main_coref_nlp = spacy.load(nlp_model_name)
          _main_coref_nlp.add_pipe("coreferee")
     return _main_coref_nlp

def get_main_st_model():
     global _main_st_model
     if _main_st_model is None:
          _main_st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
     return _main_st_model

def define_node_edge_opt(documents_list, edge_similarity_threshold=0.6):
     sentBERT_model = get_main_st_model()
     docs_sents_gen = generator_split_sentences(documents_list)
     docs_kws_gen = generator_extract_keywords(documents_list, sentBERT_model)
     docs_corfs_gen = generator_coref_resolve(documents_list)
     
     tasks = []
     for training_idx, (docs_sent_objs, docs_kws, docs_corefs) in enumerate(
               zip_longest(docs_sents_gen, docs_kws_gen, docs_corfs_gen)):
          
          if not (docs_sent_objs and docs_kws and docs_corefs):
               break
          
          # Serialize spaCy spans for consistent inter-process communication.
          sent_data = [SpanSerializer.serialize_spans(doc_sents) for doc_sents in docs_sent_objs]
          
          tasks.append((
               training_idx,
               sent_data,
               docs_kws,
               docs_corefs,
               edge_similarity_threshold,
          ))

     stop_event = threading.Event()
     monitor_thread = threading.Thread(target=monitor_usage, args=(3, stop_event))
     monitor_thread.start()
     
     cpu_num = auto_workers()
     print(f"working on {cpu_num} cpus.")
     with concurrent.futures.ProcessPoolExecutor(
          max_workers=cpu_num,
          initializer=init_subprocess,
          mp_context=multiprocessing.get_context('spawn')
     ) as executor:
          # Submit tasks in parallel.
          futures = [executor.submit(process_single_sample, task) for task in tasks]
          results = []

          for future in concurrent.futures.as_completed(futures):
               result = future.result()
               if result is not None:
                    results.append(result)
     
     stop_event.set()
     monitor_thread.join()
     
     results.sort(key=lambda x: x[0])
     sorted_results = [r[1:] for r in results]
     word_node_list, sent_node_list, edge_data_list, sentId_nodeId_list = zip(*sorted_results)
     
     del sentBERT_model
     clean_memory()
     
     return list(word_node_list), list(sent_node_list), list(edge_data_list), list(sentId_nodeId_list)

## define sub_process model
_subprocess_coref_nlp = None
_subprocess_st_model = None
def init_subprocess():
     """model initialization for each process"""
     global _subprocess_coref_nlp, _subprocess_st_model
     _subprocess_coref_nlp = spacy.load(nlp_model_name)
     _subprocess_coref_nlp.add_pipe("coreferee")
     
     _subprocess_st_model = SentenceTransformer(sent_model_name, device=device)

def process_single_sample(args):
     """Process singal sample(in one process)"""
     training_idx, sent_data, docs_kws_scores, docs_corefs, edge_threshold = args
     try:
          # init
          coref_nlp = _subprocess_coref_nlp
          st_model = _subprocess_st_model
               
          # rebuild spaCy spans
          docs_sent_objs = [SpanSerializer.deserialize_spans(sd, coref_nlp) for sd in sent_data]
          
          node_index = 0
          sentId_nodeId_map = {}
          token_node_map = {}
          sent_nodeId_map = defaultdict(list)
          
          for doc_idx, sent_objs in enumerate(docs_sent_objs):
               token_node_list = [-1] * sum(len(sent) for sent in sent_objs)
               for sent_id, sent_obj in enumerate(sent_objs):
                    sentId_nodeId_map[(training_idx, doc_idx, sent_id)] = node_index
                    sent_nodeId_map[(training_idx, doc_idx, sent_obj.text)].append(node_index)
                    
                    for token in sent_obj:
                         token_node_list[token.i] = node_index
                    node_index += 1
               
               token_node_map[(training_idx, doc_idx)] = token_node_list
          
          word_nodeId_map = {}
          word_index = node_index
          for doc_idx, doc_kws_scs in enumerate(docs_kws_scores):
               for keyword, score in doc_kws_scs:
                    if keyword not in word_nodeId_map:
                         word_nodeId_map[keyword] = word_index
                         word_index += 1
          
          edge_data = defaultdict(list)
          
          def add_edge(node1_idx, node2_idx, edge_type, weight=1.0):
               if torch.is_tensor(weight):
                    weight = weight.detach().item()
               
               if node1_idx == node2_idx: return
               
               ## advoid duplicate
               if ((node1_idx, node2_idx) in edge_data
                    and {'type': edge_type, 'weight': weight} in edge_data[(node1_idx, node2_idx)]):
                    return
               
               edge_data[(node1_idx, node2_idx)].append({'type': edge_type, 'weight': weight})
               
          
          # 1. word-sentence
          for doc_idx, sent_objs in enumerate(docs_sent_objs):
               for sent_id, sent_obj in enumerate(sent_objs):
                    sent_low = sent_obj.text.lower()
                    for word, score in docs_kws_scores[doc_idx]:
                         if word.lower() in sent_low:
                              word_node = word_nodeId_map[word]
                              sent_node = sentId_nodeId_map[(training_idx, doc_idx, sent_id)]
                              add_edge(word_node, sent_node, "word_sent", score)
          
          # 2. pronoun-antecedent
          for doc_idx, doc_corefs in enumerate(docs_corefs): ## as [(training_id, doc_id, token_id)...]
               doc_token_node_map = token_node_map[(training_idx, doc_idx)]
               for corf_cluster in doc_corefs:
                    antecedent = doc_token_node_map[corf_cluster[0][2]]
                    for i in range(1, len(corf_cluster)):
                         ## edge on each coref. the first one is the resolve
                         add_edge(doc_token_node_map[corf_cluster[i][2]], antecedent, "pronoun_antecedent")
               
          # 3. similarity
          for doc_idx, sent_objs in enumerate(docs_sent_objs):
               sents = [sent.text for sent in sent_objs]
               # chunk embedding
               chunk_size = 100
               sent_embeddings = []
               sent_size = len(sents)
               with torch.no_grad():
                    for i in range(0, sent_size, chunk_size):
                         chunk = sents[i:i+chunk_size]
                    
                         embeddings = st_model.encode(chunk, convert_to_tensor=True)
                         sent_embeddings.append(embeddings)
                    sent_embeddings = torch.cat(sent_embeddings)
               
               n = len(sent_embeddings)
               normalized = sent_embeddings / sent_embeddings.norm(dim=1, keepdim=True).to(device)
               
               for i in range(n):
                    for j in range(i+1, n):
                         similarity = (normalized[i] * normalized[j]).sum()
                         if similarity >= edge_threshold:
                              node_i = sentId_nodeId_map[(training_idx, doc_idx, i)]
                              node_j = sentId_nodeId_map[(training_idx, doc_idx, j)]
                              
                              add_edge(node_i, node_j, "similarity", weight=similarity)

               del sent_embeddings, normalized
          
          return (training_idx, word_nodeId_map, sent_nodeId_map, edge_data, sentId_nodeId_map)
     
     except Exception as e:
          print(f"[ERROR][Process {training_idx}] {str(e)}")
          traceback.print_exc()
          return None
     
     finally:
          del docs_sent_objs, edge_data
          clean_memory()

## for parallel process
class SpanSerializer:
     @staticmethod
     def serialize_spans(doc_sents):
          if not doc_sents:
               return []
          doc = doc_sents[0].doc
          return [
               (span.start_char, span.end_char, doc.text)
               for span in doc_sents
          ]

     @staticmethod
     def deserialize_spans(serialized_data, nlp_model):
          if not serialized_data:
               return []
          # get entire doc
          doc_text = serialized_data[0][2]
          doc = nlp_model(doc_text)
          
          return [
               doc.char_span(start, end, alignment_mode="expand")
               for (start, end, _) in serialized_data
          ]

def generator_split_sentences(documents_list):
     nlp = get_main_coref_nlp()
     for docs in documents_list:
          input_texts = [doc[0].strip() for doc in docs]
          docs_sents_list = []
          for doc in nlp.pipe(input_texts, batch_size=5):
               docs_sents_list.append(list(doc.sents))
          yield docs_sents_list
          del docs_sents_list
          clean_memory()
          
def generator_coref_resolve(documents_list):
     nlp = get_main_coref_nlp()
     for training_id, docs in enumerate(documents_list):
          coref_docs = []
          doc_texts = [doc[0].strip() for doc in docs]
          
          for doc_id, doc in enumerate(nlp.pipe(doc_texts, batch_size=5)):
               coref_doc = []
               
               for chain in doc._.coref_chains:
                    cluster = []
                    
                    # Process antecedent first
                    antecedent_pos = chain.most_specific_mention_index
                    antecedent = chain[antecedent_pos][0] ## token position.
                    key = (training_id, doc_id, antecedent)
                    cluster.append(key)

                    # Process other mentions
                    for idx, mention in enumerate(chain):
                         if idx == antecedent_pos:
                              continue
                         token_id = mention[0]
                         key = (training_id, doc_id, token_id)
                         cluster.append(key)

                    if len(cluster) > 1:
                         coref_doc.append(cluster)
          
               coref_docs.append(coref_doc)
          
          yield coref_docs
          
          del coref_docs, doc_texts
          clean_memory()

def auto_workers():
     mem_available = psutil.virtual_memory().available / (1024 ** 3)  # GB
     model_mem = 2  # astimated mem usage for each process
     cpu_cnt = multiprocessing.cpu_count()
     return min(cpu_cnt + 1, max(1, int(mem_available // model_mem)))

def monitor_usage(interval, stop_event):
     while not stop_event.is_set():
          print_cpu_memory(label="multiprocess", interval=interval)
          time.sleep(60 * 5) ## every 5 min
          