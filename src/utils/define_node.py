import os
import concurrent.futures
import multiprocessing
import threading
import traceback
from collections import defaultdict
from keybert import KeyBERT
import string
from spacy.matcher import PhraseMatcher
import torch
import torch.nn.functional as F
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import traceback
import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
import numpy as np
import hdbscan


from utils.model_utils import clean_memory, monitor_usage, auto_workers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NLP_MODEL_NAME = "en_core_web_lg"
SENT_MODEL_NAME = "all-mpnet-base-v2"
WORKER_NLP_BATCH_SIZE = 100 # for spacy docs process
WORKER_TASK_BATCH_SIZE = 10 # for each subprocess

# These will hold the models loaded *within each subprocess*
_subprocess_coref_nlp = None
_subprocess_st_model = None
_subprocess_keyword_model = None

def init_subprocess():
     """Initialize models within each worker process."""
     # faiss.omp_set_num_threads(max(auto_workers() // 2, 2))
     
     global _subprocess_coref_nlp, _subprocess_st_model, _subprocess_keyword_model

     if _subprocess_coref_nlp is None:
          _subprocess_coref_nlp = spacy.load(NLP_MODEL_NAME)
          if "coreferee" not in _subprocess_coref_nlp.pipe_names:
               _subprocess_coref_nlp.add_pipe("coreferee")

     if _subprocess_st_model is None:
          _subprocess_st_model = SentenceTransformer(SENT_MODEL_NAME, device=device)
          _subprocess_st_model.eval()
          # _subprocess_st_model = _subprocess_st_model.to(device)

     if _subprocess_keyword_model is None:
          _subprocess_keyword_model = KeyBERT(model=_subprocess_st_model)

def cleanup_subprocess():
     global _subprocess_coref_nlp, _subprocess_st_model, _subprocess_keyword_model

     del _subprocess_coref_nlp
     del _subprocess_st_model
     del _subprocess_keyword_model

     _subprocess_coref_nlp = None
     _subprocess_st_model = None
     _subprocess_keyword_model = None

     clean_memory()
          

def compute_edges_similarity(sents, edge_threshold, encode_batch_size=256, sim_batch_size=1024):
     if not sents:
          return [], None

     embeddings = []
     with torch.inference_mode():
          for i in range(0, len(sents), encode_batch_size):
               batch = sents[i:i + encode_batch_size]
               emb = _subprocess_st_model.encode(batch, convert_to_tensor=True, device=device, show_progress_bar=False)
               embeddings.append(emb)

     if not embeddings:
          return [], None

     embeddings = torch.cat(embeddings)
     normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

     edges = []
     n_sents = len(normalized_embeddings)

     for i in range(0, n_sents, sim_batch_size):
          batch_i = normalized_embeddings[i:min(i + sim_batch_size, n_sents)]

          # Calculate similarities within the batch (upper triangle)
          sims_intra = torch.mm(batch_i, batch_i.T)
          rows, cols = torch.triu_indices(sims_intra.size(0), sims_intra.size(1), offset=1, device=device) # Create indices on device

          if rows.numel() > 0:
               sim_values_intra = sims_intra[rows, cols]
               mask = sim_values_intra >= edge_threshold
               valid_rows = rows[mask]
               valid_cols = cols[mask]
               valid_sims = sim_values_intra[mask]
               for r, c, sim_val in zip(valid_rows, valid_cols, valid_sims):
                    gi, gj = i + r.item(), i + c.item() # Global indices
                    edges.append((gi, gj, sim_val.item()))
               del valid_rows, valid_cols, valid_sims, mask, sim_values_intra

          del rows, cols, sims_intra

          # Calculate similarities with subsequent batches
          for j in range(i + sim_batch_size, n_sents, sim_batch_size):
               j_end = min(j + sim_batch_size, n_sents)
               batch_j = normalized_embeddings[j:j_end]

               sims_inter = torch.mm(batch_i, batch_j.T)
               mask_inter = sims_inter >= edge_threshold
               idx = torch.nonzero(mask_inter)

               if idx.numel() > 0:
                    sim_values_inter = sims_inter[idx[:, 0], idx[:, 1]]
                    for (loc_i, loc_j), sim_val in zip(idx, sim_values_inter):
                         gi, gj = i + loc_i.item(), j + loc_j.item() # Global indices
                         edges.append((gi, gj, sim_val.item()))
                    del sim_values_inter

               del sims_inter, mask_inter, idx, batch_j

          del batch_i

     return edges

def process_batch(batch_input):
     """
     Processes a batch of documents within a single worker process,
     using the input format: [(sample_docs, threshold)].
     Groups processing by original sample index to maintain contiguous node IDs per sample.

     Args:
          batch_input (tuple): A tuple containing:
               - sample_docs (list[tuple[int, int, str]]):
                    List of documents, each as (original_training_idx, doc_idx_in_sample, doc_text).
               - edge_similarity_threshold (float): The threshold for similarity edges for this batch.

     Returns:
          list[map] * 4: A list of results, one tuple per original sample index present in the batch.
                         Each tuple: sample_word_nodeId_map_list,
                                        sample_sent_nodeId_map_by_key_list, sample_edge_data_list,
                                        sample_sentId_nodeId_map
                         Returns None if a critical error occurs.
     """
     global _subprocess_coref_nlp, _subprocess_st_model, _subprocess_keyword_model

     try:
          if not batch_input: return []
          edge_similarity_threshold = batch_input[0][1]
          list_of_docs = [doc for sample, _ in batch_input for doc in sample] ## ([(training_id, doc_id, doc_text), (),...], threshold)
          if not isinstance(list_of_docs, list) or not isinstance(edge_similarity_threshold, (float, int)):
               raise ValueError("Invalid batch_input structure")
     except (TypeError, ValueError) as e:
          print(f"[ERROR][Worker {os.getpid()}] Invalid input format: {e}")
          traceback.print_exc()
          return None


     batch_results_by_sample = []
     samples_in_batch = defaultdict(list)
     all_doc_texts = []
     doc_identifiers = []

     if not list_of_docs:
          return []

     for i, doc_tuple in enumerate(list_of_docs):
          try:
               original_idx, doc_idx, doc_text = doc_tuple
               if not isinstance(original_idx, int) or not isinstance(doc_idx, int) or not isinstance(doc_text, str):
                    print(f"[WARN] Skipping invalid item in list_of_docs at index {i}: {doc_tuple}")
                    continue
               
               compressed_text = compact_text(doc_text, eps=0.50, min_samples=2, EMB_BATCH_SIZE=16)
               # compressed_text = compact_text_auto_kmeans(doc_text, EMB_BATCH_SIZE=16, min_k=1, max_k=10)
               all_doc_texts.append(compressed_text)
               doc_identifiers.append((original_idx, doc_idx))
               samples_in_batch[original_idx].append({'doc_idx_in_sample': doc_idx, 'batch_list_index': i})
          except (TypeError, ValueError):
               traceback.print_exc()
               raise


     if not all_doc_texts: # all items were invalid
          return []

     # spaCy Processing
     processed_docs_ordered = list(_subprocess_coref_nlp.pipe(all_doc_texts, batch_size=WORKER_NLP_BATCH_SIZE))
     
     # Keyword Extraction
     docs_kws_scores_ordered = extract_keywords_internal(all_doc_texts)

     #  Process each ORIGINAL SAMPLE group ---
     for original_training_idx, sample_items in samples_in_batch.items():

          current_sample_node_offset = 0
          sample_word_nodeId_map = {}
          sample_sent_nodeId_map_by_key = defaultdict(list)
          sample_sentId_nodeId_map = {}
          sample_token_node_map_by_doc = {}
          sample_doc_sents_map = defaultdict(list)
          sample_edge_data = defaultdict(list)

          sample_all_sent_texts = []
          sample_sent_index_to_node_id = []
          
          # Sort items by doc_idx_in_sample to ensure consistent processing order within the sample
          sample_items.sort(key=lambda x: x['doc_idx_in_sample'])

          for item_info in sample_items:
               doc_idx_in_sample = item_info['doc_idx_in_sample']
               batch_list_index = item_info['batch_list_index'] # Index in the original all_doc_texts list

               doc = processed_docs_ordered[batch_list_index]
               doc_sents = list(doc.sents)
               
               doc_node = current_sample_node_offset
               current_sample_node_offset += 1
               
               # --- Assign sentence node IDs (contiguous within doc) ---
               doc_token_to_sample_node_list = [-1] * len(doc)
               for sent_id_in_doc, sent_obj in enumerate(doc_sents):
                    current_sent_node_id_in_sample = current_sample_node_offset
                    sent_text = sent_obj.text
                    sample_sentId_nodeId_map[(original_training_idx, doc_idx_in_sample, sent_id_in_doc)] = current_sent_node_id_in_sample
                    sample_sent_nodeId_map_by_key[(original_training_idx, doc_idx_in_sample, sent_text)].append(current_sent_node_id_in_sample)

                    for token in sent_obj:
                         if 0 <= token.i < len(doc_token_to_sample_node_list):
                              doc_token_to_sample_node_list[token.i] = current_sent_node_id_in_sample

                    sample_all_sent_texts.append(sent_text)
                    sample_sent_index_to_node_id.append(current_sent_node_id_in_sample)
                    
                    # 1. Doc-Sent edge
                    _add_edge_local(sample_edge_data, doc_node, current_sent_node_id_in_sample, "doc_sent", 1.0)
                    sample_doc_sents_map[doc_node].append(current_sent_node_id_in_sample)
                    
                    current_sample_node_offset += 1 # Increment offset for the next sentence in the sample

               sample_token_node_map_by_doc[(original_training_idx, doc_idx_in_sample)] = doc_token_to_sample_node_list

               # 2. Coreference Edges
               if doc._.coref_chains:
                    doc_token_map = sample_token_node_map_by_doc.get((original_training_idx, doc_idx_in_sample), [])
                    if not doc_token_map:
                         print(f"[WARN] Sample {original_training_idx} has no coreference token mapping.")
                         continue

                    for chain in doc._.coref_chains:
                         antecedent_node = -1
                         try:
                              antecedent_mention_index = chain.most_specific_mention_index
                              antecedent_token_index = chain[antecedent_mention_index][0]
                              if 0 <= antecedent_token_index < len(doc_token_map):
                                   antecedent_node = doc_token_map[antecedent_token_index] # Gets SAMPLE-level node ID
                         except IndexError:
                              traceback.print_exc()
                              raise

                    if antecedent_node is None or antecedent_node < 0: 
                         print(f"[Warning] Sample {original_training_idx} sentences have no antecedent. ")
                         continue

                    for mention_idx, mention in enumerate(chain):
                         if mention_idx == antecedent_mention_index: continue
                         pronoun_node = -1
                         try:
                              pronoun_token_index = mention[0]
                              if 0 <= pronoun_token_index < len(doc_token_map):
                                   pronoun_node = doc_token_map[pronoun_token_index] # Gets SAMPLE-level node ID
                         except IndexError:
                              traceback.print_exc()
                              raise

                         if pronoun_node is not None and pronoun_node >= 0 and pronoun_node != antecedent_node:
                              _add_edge_local(sample_edge_data, pronoun_node, antecedent_node, "pronoun_antecedent", 1.0)

          
          # 3. Similarity Edges
          similarity_edges = compute_edges_similarity_ann(sample_all_sent_texts, edge_similarity_threshold)
          # if similarity_edges is None or len(similarity_edges) == 0:
          #      print(f"[Warning] Sample {original_training_idx} has no similarity edges.")
          
          for node_i_local, node_j_local, sim_value in similarity_edges:
               sample_level_node_i = sample_sent_index_to_node_id[node_i_local]
               sample_level_node_j = sample_sent_index_to_node_id[node_j_local]
               _add_edge_local(sample_edge_data, sample_level_node_i, sample_level_node_j, "similarity", weight=sim_value)


          # --- Assign word node IDs for the ENTIRE SAMPLE ---
          kws_scores_map = defaultdict(list)

          for item_info in sample_items:
               batch_list_index = item_info['batch_list_index']
               if batch_list_index < len(docs_kws_scores_ordered):
                    doc_kws_scores = docs_kws_scores_ordered[batch_list_index]
                    for keyword, score in doc_kws_scores:
                         kws_scores_map[keyword].append(score)

          kws_scores_node_map = {}
          phrase_matcher = PhraseMatcher(_subprocess_coref_nlp.vocab, attr='LOWER')
          
          for keyword, score_list in kws_scores_map.items():
               score_avg = sum(score_list) / len(score_list) if score_list else 0.0
               sample_word_nodeId_map[keyword] = current_sample_node_offset
               kws_scores_node_map[keyword] = (current_sample_node_offset, score_avg) # store node ID and average score
               
               phrase_matcher.add(f"{keyword}",[_subprocess_coref_nlp(keyword)]) # Add keyword to matcher
               
               current_sample_node_offset += 1

          # 4. Word-Sentence Edges for the ENTIRE SAMPLE
          for item_info in sample_items:
               doc_idx_in_sample = item_info['doc_idx_in_sample']
               batch_list_index = item_info['batch_list_index']

               doc = processed_docs_ordered[batch_list_index]
               
               matches = phrase_matcher(doc)
               for match_id, start, end in matches:
                    matched_keyword = _subprocess_coref_nlp.vocab.strings[match_id]
                    if matched_keyword in kws_scores_node_map:
                         node_id, score = kws_scores_node_map[matched_keyword]
                         sent_id = sample_token_node_map_by_doc[(original_training_idx, doc_idx_in_sample)][start]
                         
                         _add_edge_local(sample_edge_data, node_id, sent_id, "word_sent", score)
               

          batch_results_by_sample.append((
               original_training_idx,
               sample_word_nodeId_map,
               sample_sent_nodeId_map_by_key,
               sample_edge_data,
               sample_sentId_nodeId_map,
               sample_doc_sents_map,
          ))

          del sample_word_nodeId_map, sample_sent_nodeId_map_by_key, sample_edge_data
          del sample_sentId_nodeId_map, sample_token_node_map_by_doc, sample_doc_sents_map,
     
     return batch_results_by_sample

def _add_edge_local(edge_data_dict, node1_idx, node2_idx, edge_type, weight=1.0):
     """Adds an edge to the provided dictionary, handling sorting and weights."""
     if node1_idx is None or node2_idx is None or node1_idx < 0 or node2_idx < 0:
          print(f"[WARN] Invalid node index encountered: {node1_idx}, {node2_idx}. Skipping edge.")
          return

     weight = float(weight.item()) if torch.is_tensor(weight) else float(weight)

     if node1_idx == node2_idx: return

     # Use sorted tuple for consistent key order, ensures edge (a,b) is same as (b,a)
     key = tuple(sorted((node1_idx, node2_idx)))

     current_edges = edge_data_dict.get(key, [])
     for edge in current_edges:
          if edge['type'] == edge_type:
               return

     edge_data_dict[key].append({'type': edge_type, 'weight': weight})

def compact_text(doc_text, eps=0.5, min_samples=2, EMB_BATCH_SIZE=32):
     """
     Compacts a document by clustering sentence embeddings and selecting sentences from core clusters.

     Args:
          doc_text (str): The input text document.
          min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
                              A lower value allows for less dense clusters.
          EMB_BATCH_SIZE (int): The batch size for sentence embedding computation.

     Returns:
          str: The compacted text or the original text if no clusters are found.
     """
     global _subprocess_coref_nlp, _subprocess_st_model

     doc = _subprocess_coref_nlp(doc_text)
     original_sentence_texts = [s.text.strip() for s in doc.sents if s.text.strip()]
     
     if not original_sentence_texts:
          return ""

     if len(original_sentence_texts) <= 5:
          return doc_text

     all_embeddings = _subprocess_st_model.encode(
          original_sentence_texts,
          convert_to_tensor=True,
          device=device,
          batch_size=EMB_BATCH_SIZE,
          cluster_selection_epsilon=eps,
          show_progress_bar=False
     )
     all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
     sentence_embeddings_np = all_embeddings.cpu().numpy()
     
     num_sentences = len(original_sentence_texts)
     min_cluster_size = max(2, min(5, int(num_sentences * 0.2)))

     clusterer = hdbscan.HDBSCAN(
          min_cluster_size=min_cluster_size,
          min_samples=min_samples,
          metric='euclidean',
          cluster_selection_method='eom',
          allow_single_cluster=True,
     )
     cluster_labels = clusterer.fit_predict(sentence_embeddings_np)
     kept_sentences = [s for s, label in zip(original_sentence_texts, cluster_labels) if label != -1]

     return " ".join(kept_sentences) if kept_sentences else doc_text

def count_words(text):
     """Counts words in a text after removing punctuation."""
     if not isinstance(text, str): # Handle potential non-string input
          return 0
     translator = str.maketrans('', '', string.punctuation)
     text_without_punct = text.translate(translator)
     words = text_without_punct.split()
     return len(words)

def extract_keywords_internal(docs_text, words_per_100=1, min_keywords=1, max_keywords=15):
     global _subprocess_keyword_model

     if not docs_text or min_keywords <= 0 or max_keywords <= 0 or words_per_100 <= 0 or max_keywords <= min_keywords:
          return []

     if _subprocess_keyword_model is None:
          print("[ERROR] Keyword model not initialized in subprocess!")
          return [[] for _ in docs_text]


     # 1. Calculate desired top_n for EACH document
     specific_top_n_list = []
     for text in docs_text:
          doc_word_count = count_words(text)
          desired_n = max(min_keywords, min(max_keywords, (doc_word_count // 100) * words_per_100))

          specific_top_n_list.append(desired_n)

     # 2. Find the maximum top_n needed in this batch
     max_top_n_needed = max(specific_top_n_list + [min_keywords])
     max_top_n_needed = min(max_top_n_needed, max_keywords)

     if max_top_n_needed == 0:
          return [[] for _ in docs_text]

     # 3. Run batch extraction asking for the maximum needed top_n
     try:
          batch_keywords_raw = _subprocess_keyword_model.extract_keywords(
               docs_text,
               top_n=max_top_n_needed,
               stop_words='english',
               use_mmr=True,
               diversity=0.6,
               keyphrase_ngram_range=(1, 1)
          )

     except Exception as e:
          print(f"[ERROR][Keyword Extraction] Batch extraction failed: {e}")
          traceback.print_exc()
          return [[] for _ in docs_text]

     # 4. Filter results down to the specific top_n for each document
     final_keywords = []
     if len(batch_keywords_raw) != len(specific_top_n_list):
          print(f"[ERROR] Mismatch between keyword results ({len(batch_keywords_raw)}) and input docs ({len(specific_top_n_list)})")
          return [[] for _ in docs_text]

     for i, specific_n in enumerate(specific_top_n_list):
          # Get the raw keywords for this document (it's already sorted by relevance)
          doc_raw_keywords = batch_keywords_raw[i]
          final_keywords.append(doc_raw_keywords[:specific_n])

     return final_keywords

def compute_edges_similarity_ann(sentence_texts, threshold, EMB_BATCH_SIZE=32):
     """
     Computes similarity edges using FAISS KNN search and filtering based on
     the threshold value of cosine similarity. ).

     Encodes input sentence texts internally using a pre-loaded model.
     Automatically attempts to use GPU if faiss-gpu is installed and CUDA is available.
     """
     global _subprocess_st_model # Ensure model is accessible

     if not sentence_texts or len(sentence_texts) < 2:
          return []

     embeddings_tensor = None
     try:
          with torch.inference_mode(), torch.cuda.amp.autocast():
               embeddings_tensor = _subprocess_st_model.encode(
                    sentence_texts,
                    convert_to_tensor=True,
                    device='cpu',
                    show_progress_bar=False,
                    batch_size=EMB_BATCH_SIZE
               )
               
               embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
     except Exception as e:
          print(f"[ERROR][Worker {os.getpid()}] Sentence encoding failed: {e}")
          traceback.print_exc()
          return []

     if embeddings_tensor is None or embeddings_tensor.shape[0] < 2:
          print(f"[WARN][Worker {os.getpid()}] Invalid embeddings tensor shape after encoding.")
          return []

     embeddings_np = embeddings_tensor.cpu().numpy().astype('float32')
     embeddings_tensor = np.ascontiguousarray(embeddings_tensor) # to cpu
     n_sents, dim = embeddings_np.shape
     
     index = faiss.IndexFlatIP(dim)
     edges = []
     try:
          index.add(embeddings_np)
          k = min(64, n_sents)
          batch_size_search = min(32, n_sents)
          D_list = []
          I_list = []

          for i_start in range(0, n_sents, batch_size_search):
               i_end = min(i_start + batch_size_search, n_sents)
               batch_query = embeddings_np[i_start:i_end]
               D_batch, I_batch = index.search(batch_query, k=k)
               D_list.append(D_batch)
               I_list.append(I_batch)

          D = np.vstack(D_list)
          I = np.vstack(I_list)
          # D contains cosine similarities (scores)
          # I contains the indices of the neighbors

          added_pairs = set()
          for i in range(n_sents):
               for neighbor_rank in range(k):
                    j = I[i, neighbor_rank]
                    sim = D[i, neighbor_rank]
                    
                    # Skip invalid results or self-comparison
                    if j == -1 or i == j:
                         continue
                    
                    if sim >= threshold:
                         pair = tuple(sorted((i, j)))
                         if pair not in added_pairs:
                              edges.append((i, j, float(sim)))
                              added_pairs.add(pair)
                              
     except Exception as e:
          print(f"[ERROR][Worker {os.getpid()}] FAISS search or processing failed: {e}")
          traceback.print_exc()
          return []
     
     return edges


def define_node_edge_opt_parallel(documents_list, edge_similarity_threshold=0.6):
     """
     Optimized function to define nodes and edges using parallel processing
     for the entire pipeline (preprocessing + node/edge creation).

     Args:
          documents_list (list): List of samples, where each sample is typically
                                   a list of document string, e.g., [[[text1], [text2]], ...]
          edge_similarity_threshold (float): Threshold for similarity edges.

     Returns:
          tuple: (all_word_nodes, all_sent_nodes, all_edge_data, all_sentId_maps)
                    Aggregated results across all documents. Node IDs need careful handling.
                    Returning structure might need adjustment based on downstream use.
     """
     # --- Prepare tasks for parallel execution ---
     # Each task will be a batch of documents to process.
     tasks = []
     len_docs = len(documents_list)

     for i in range(0, len_docs, WORKER_TASK_BATCH_SIZE):
          end_idx = min(i + WORKER_TASK_BATCH_SIZE, len_docs)

          batch = [
               (
                    [(j, doc_id, doc[0]) for doc_id, doc in enumerate(documents_list[j])],
                    edge_similarity_threshold
               )
               for j in range(i, end_idx)
          ]

          tasks.append(batch)

     cpu_num = auto_workers()
     all_results_flat = [] # Store results from all workers

     # Start memory monitor
     stop_event = threading.Event()
     monitor_thread = threading.Thread(target=monitor_usage, args=(3, stop_event, "processing sample"))
     monitor_thread.start()

     with concurrent.futures.ProcessPoolExecutor(
          max_workers=cpu_num,
          initializer=init_subprocess,
          mp_context= multiprocessing.get_context('spawn')
     ) as executor:
          futures = [executor.submit(process_batch, task_batch) for task_batch in tasks]

          for future in tqdm(
               concurrent.futures.as_completed(futures),
               total=len(futures),
               desc="Graph Node-Edge Batch-Processing",
               position=0
          ):
               if future.exception() is not None:
                    print("\n[ERROR] Worker crashed with traceback:")
                    traceback.print_exception(
                         type(future.exception()),
                         future.exception(),
                         future.exception().__traceback__
                    )
                    continue
               
               try:
                    batch_result = future.result()
                    if batch_result is not None:
                         all_results_flat.extend(batch_result)
                    else:
                         print("[WARN] A worker batch failed processing.")
               except Exception as e:
                    print(f"[ERROR] Fetching result from future failed: {e}")
                    traceback.print_exc()

     # Stop monitoring
     stop_event.set()
     monitor_thread.join()

     cleanup_subprocess()
     
     # --- Aggregate Results ---
     all_results_flat.sort(key=lambda x: x[0]) # Sort by original_training_idx

     word_node_list = []
     sent_node_list = []
     edge_data_list = []
     sentId_nodeId_list = []
     doc_sents_list = []
     
     for sample_res in all_results_flat:
          original_training_idx, sample_word_nodeId_map, sample_sent_nodeId_map_by_key, sample_edge_data, sample_sentId_nodeId_map, sample_doc_sents_map = sample_res
          
          # Append to global lists
          word_node_list.append(sample_word_nodeId_map)
          sent_node_list.append(sample_sent_nodeId_map_by_key)
          edge_data_list.append(sample_edge_data)
          sentId_nodeId_list.append(sample_sentId_nodeId_map)
          doc_sents_list.append(sample_doc_sents_map)

     del all_results_flat, tasks, futures
     clean_memory()

     return word_node_list, sent_node_list, edge_data_list, sentId_nodeId_list, doc_sents_list

def compact_text_auto_kmeans(doc_text, EMB_BATCH_SIZE=32):
     global _subprocess_coref_nlp, _subprocess_st_model

     doc = _subprocess_coref_nlp(doc_text)
     original_sentence_texts = [s.text.strip() for s in doc.sents if s.text.strip()]
     num_sentences = len(original_sentence_texts)

     if num_sentences <= 3:
          return doc_text

     with torch.inference_mode(), torch.no_grad():
          all_embeddings = _subprocess_st_model.encode(
               original_sentence_texts, convert_to_tensor=True,
               device='cuda' if torch.cuda.is_available() else 'cpu',
               batch_size=EMB_BATCH_SIZE
          )
     sentence_embeddings_np = F.normalize(all_embeddings, p=2, dim=1).cpu().numpy()

     min_k = 3
     max_k = min(12, int(num_sentences * 0.3))

     if min_k >= max_k:
          optimal_k = min(min_k, num_sentences)
     else:
          silhouette_scores = []
          k_range = range(min_k, max_k + 1)
          for k in k_range:
               kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
               labels = kmeans.fit_predict(sentence_embeddings_np)
               if len(set(labels)) > 1:
                    score = silhouette_score(sentence_embeddings_np, labels)
               silhouette_scores.append(score)
          
          if not silhouette_scores: return doc_text
          optimal_k = k_range[np.argmax(silhouette_scores)]
          
     kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
     labels = kmeans.fit(sentence_embeddings_np)
     
     representative_indices = []
     for i in range(optimal_k):
          cluster_indices = np.where(labels.labels_ == i)[0]
          if len(cluster_indices) == 0: continue
          cluster_embeddings = sentence_embeddings_np[cluster_indices]
          distance_matrix = pairwise_distances(cluster_embeddings, metric='cosine')
          medoid_index_in_cluster = np.argmin(distance_matrix.sum(axis=1))
          medoid_original_index = cluster_indices[medoid_index_in_cluster]
          representative_indices.append(medoid_original_index)

     kept_sentences = [original_sentence_texts[i] for i in sorted(representative_indices)]
     
     return " ".join(kept_sentences)