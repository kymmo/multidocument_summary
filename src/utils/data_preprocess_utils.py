import json
import spacy
from keybert import KeyBERT
import coreferee
import torch
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from concurrent.futures import ProcessPoolExecutor
from functools import wraps
import traceback
import os

from utils.model_utils import clean_memory, print_gpu_memory

# Load models - this should be done only once
nlp_coref = spacy.load("en_core_web_lg")
nlp_coref.add_pipe('coreferee')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_jsonl(file_path):
     """  Loads data from a JSONL file with the correct multi-document format.
          Each line represents one sample to multi-doc summarization
     """
     documents_list = []
     summary_list = []
     with open(file_path, 'r', encoding='utf-8') as f:
          for line in f:
               try:
                    data = json.loads(line)
                    if len(data['document']) > 1: # except single doc
                         documents_list.append(data['document']) # Each line is a list of documents
                         summary_list.append(data['summary'])
               except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
     
     return documents_list, summary_list

def split_sentences_pipe(documents_list):
     """Splits sentences in each document within the list.
     return sent object list.
     """
     processed_documents_list = []
     
     for docs in documents_list:
          input_texts = [doc[0].strip() for doc in docs]
          docs_sents_list = [
               [sent for sent in doc.sents]
               for doc in nlp_coref.pipe(input_texts, batch_size=5)
          ]
          processed_documents_list.append(docs_sents_list)
     
     return processed_documents_list

def parallel_error_handler(default_output=None, log_errors=True):
     def decorator(func):
          @wraps(func)
          def wrapper(*args, **kwargs):
               try:
                    return func(*args, **kwargs)
               except Exception as e:
                    if log_errors:
                         error_msg = f"[PID:{os.getpid()}] {func.__name__} FAIL. Input: {str(args)[:50]}...\n{traceback.format_exc()}"
                         print(error_msg)
                    return default_output
          return wrapper
     
     return decorator

def split_sentences2(documents_list):
     """_summary_ higher speed

     Args:
          documents_list (_type_): original doc list

     Returns:
          _type_: _description_
     """
     processed_documents_list = []
     
     with ProcessPoolExecutor(max_workers= max(1, os.cpu_count()//2)) as executor:
          for docs in documents_list:
               input_texts = [doc[0].strip() for doc in docs]
               docs_sents_list = list(executor.map(optimized_split_doc_sent, input_texts, chunksize=20))

               processed_documents_list.append(docs_sents_list)
     
     return processed_documents_list

@parallel_error_handler(default_output=[], log_errors=True)
def optimized_split_doc_sent(text):
     # load global model
     global nlp_coref
     doc = nlp_coref(text.strip())
     return [sent.text for sent in doc.sents]

def extract_keywords(documents_list,sentBERT_model, words_per_100=1, min_keywords=2, max_keywords=15):
     """extract key word from each document, default 10 words

     Args:
          documents_list (_type_): original documents sample list. No need to split document

     Returns:
          _type_: list, keywords of each document
     """
     
     keywords_list = []
     kw_model = KeyBERT(model=sentBERT_model)

     for documents in documents_list:
          ## dynamic top_n
          avg_length = sum(len(doc) for doc in documents) // len(documents)
          top_n = max(min_keywords, min(max_keywords, (avg_length // 100) * words_per_100))
          documents = [doc[0] for doc in documents] ## convert to string list
          
          keywords = kw_model.extract_keywords(
               documents,
               top_n=top_n,
               stop_words='english',
               use_mmr=True,
               diversity=0.6,
               keyphrase_ngram_range=(1, 1)
          )
          
          keywords_list.append(keywords)
     
     del kw_model
     clean_memory()
     
     return keywords_list

def coref_resolve(documents_list):
     """_summary_ coreference resolve

     Args:
          documents_list (_type_): documents with sentences

     Returns:
          _type_: coref_cluster for each document. form as orginal dataset. cluster element: (training_idx, doc_idx, sent), to find node id.
     """
     coref_docs_list= []
     for training_id, docs in enumerate(documents_list):
          coref_docs = []
          for doc_id, document in enumerate(docs):
               doc = nlp_coref(document[0].strip())
               coref_doc = []
               
               for chain in doc._.coref_chains: ## coreference cluster
                    coref_cluster = []
                    ## resolve the sentence it belongs to. antecednet in the first place
                    antecedent_pos = chain.most_specific_mention_index
                    antecedent_id = chain[antecedent_pos][0]
                    ant_sent_txt = doc[antecedent_id].sent
                    
                    coref_cluster.append((training_id, doc_id, str(ant_sent_txt)))
                    
                    for idx, mention in enumerate(chain):
                         if idx == antecedent_pos: continue
                         
                         token_id = mention[0]
                         sent_txt = doc[token_id].sent
                         key = (training_id, doc_id, str(sent_txt))
                         if key not in coref_cluster:
                              coref_cluster.append(key)

                    if len(coref_cluster) > 1: ## can not connect with itself
                         coref_doc.append(coref_cluster)
          
               coref_docs.append(coref_doc)
          
          coref_docs_list.append(coref_docs)
     
     return coref_docs_list

def coref_resolve2(documents_list):
     """ higher speed
     return (training_id, doc_id, the coreference cluster token id)
     """
     # Collect all texts and their indices for batch processing
     all_texts = []
     index_map = []  # Stores (training_id, doc_id) for each text
     for training_id, docs in enumerate(documents_list):
          for doc_id, document in enumerate(docs):
               text = document[0].strip()
               all_texts.append(text)
               index_map.append((training_id, doc_id))

     # Batch process all documents
     processed_docs = list(nlp_coref.pipe(all_texts))

     # Initialize result structure
     coref_docs_list = [
          [[] for _ in docs]
          for docs in documents_list
     ]

     # Process each document's results
     for (training_id, doc_id), doc in zip(index_map, processed_docs):
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
                    # sent = str(doc[token_id].sent)
                    key = (training_id, doc_id, token_id)
                    cluster.append(key)

               if len(cluster) > 1:
                    coref_doc.append(cluster)
          
          coref_docs_list[training_id][doc_id] = coref_doc

     return coref_docs_list

def define_node_edge(documents_list, edge_similarity_threshold = 0.6):
     """_summary_ index node, including word and sentence; define all types of edges

     Args:
          documents_list (_type_): original docs

     Returns:
          _type_:  word_node_map, sent_node_map, edge list, sentid_nodeid_map
     """
     sentBERT_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

     prepro_start_time = time.time()
     docs_sents_obj_list = split_sentences_pipe(documents_list)
     docs_kws_scores_list = extract_keywords(documents_list, sentBERT_model)
     docs_corefs_list = coref_resolve2(documents_list)
     prepro_end_time = time.time()
     print(f"Finish preprocess, time cost:  {prepro_end_time - prepro_start_time:.4f} s.")
     
     edge_data_list = []
     word_node_list = []
     sent_node_list = []
     sentId_nodeId_list = []
     for training_idx, (docs_sent_objs, docs_kws_scores, docs_corefs) in enumerate(zip(docs_sents_obj_list, docs_kws_scores_list, docs_corefs_list)):
          node_index = 0
          
          # sentence node index
          sentId_nodeId_map = {}
          token_node_map = {} ## for coreference resolve. each doc hold a token-node list
          sent_nodeId_map = defaultdict(list) ## need to handle repeat sent
          for doc_idx, sent_objs in enumerate(docs_sent_objs):
               token_node_list = [-1] * sum(len(sent) for sent in sent_objs)
               for sent_id, sent_obj in enumerate(sent_objs):
                    sentId_nodeId_map[(training_idx, doc_idx, sent_id)] = node_index
                    sent_nodeId_map[((training_idx, doc_idx, sent_obj.text))].append(node_index)
                    
                    for token in sent_obj:
                         token_node_list[token.i] = node_index
                    node_index += 1
               
               token_node_map[(training_idx, doc_idx)] = token_node_list
               
          # word node index
          word_nodeId_map = {} ## for word-sentence edge
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
                    
               ## advoid duplicate
               if node1_idx == node2_idx: return
               
               if ((node1_idx, node2_idx) in edge_data
                    and {'type': edge_type, 'weight': weight} in edge_data[(node1_idx, node2_idx)]):
                    return
               
               edge_data[(node1_idx, node2_idx)].append({'type': edge_type, 'weight': weight})
                    
          ## 1. word-sentence
          for doc_idx, sent_objs in enumerate(docs_sent_objs):
               for sent_id, sent_obj in enumerate(sent_objs):
                    ## check wether contain keywords
                    sent_low = sent_obj.text.lower()
                    for word, score in docs_kws_scores[doc_idx]:
                         if word.lower() in sent_low:
                              word_node = word_nodeId_map[word]
                              sent_node = sentId_nodeId_map[(training_idx, doc_idx, sent_id)]
                              add_edge(word_node, sent_node, "word_sent", weight=score)
                              
          ## 2. pronoun-antecedent
          for doc_idx, doc_corefs in enumerate(docs_corefs): ## as [(training_id, doc_id, token_id)...]
               doc_token_node_map = token_node_map[(training_idx, doc_idx)]
               for corf_cluster in doc_corefs:
                    antecedent = doc_token_node_map[corf_cluster[0][2]]
                    for i in range(1, len(corf_cluster)):
                         ## edge on each coref. the first one is the resolve
                         add_edge(doc_token_node_map[corf_cluster[i][2]], antecedent, "pronoun_antecedent")
          
          ## 3. similarity
          for doc_idx, sent_objs in enumerate(docs_sent_objs):
               sents = [sent.text for sent in sent_objs]
               # chunk embedding
               chunk_size = 100
               sent_embeddings = []
               sent_size = len(sents)
               with torch.no_grad():
                    for i in range(0, sent_size, chunk_size):
                         chunk = sents[i:i+chunk_size]
                    
                         embeddings = sentBERT_model.encode(chunk, convert_to_tensor=True)
                         sent_embeddings.append(embeddings)
                    sent_embeddings = torch.cat(sent_embeddings)
               
               n = len(sent_embeddings)
               normalized = sent_embeddings / sent_embeddings.norm(dim=1, keepdim=True).to(device)
               
               for i in range(n):
                    for j in range(i+1, n):
                         similarity = (normalized[i] * normalized[j]).sum()
                         if similarity >= edge_similarity_threshold:
                              node_i = sentId_nodeId_map[(training_idx, doc_idx, i)]
                              node_j = sentId_nodeId_map[(training_idx, doc_idx, j)]
                              
                              add_edge(node_i, node_j, "similarity", weight=similarity)

               del sent_embeddings, normalized
          
          edge_data_list.append(edge_data)
          word_node_list.append(word_nodeId_map)
          sent_node_list.append(sent_nodeId_map)
          sentId_nodeId_list.append(sentId_nodeId_map)
     
     del sentBERT_model
     clean_memory()
     
     print_gpu_memory("after preprocess")
     
     return word_node_list, sent_node_list, edge_data_list, sentId_nodeId_list

def chunker(seq, chunk_size):
     for i in range(0, len(seq), chunk_size):
          yield seq[i:i+chunk_size]