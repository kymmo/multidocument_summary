import json
import spacy
from keybert import KeyBERT
import coreferee
import torch
import utils.Logger as mylogger
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# Load models - this should be done only once
nlp_sm = spacy.load("en_core_web_sm")
nlp_coref = spacy.load("en_core_web_lg")
nlp_coref.add_pipe('coreferee')
kw_model = KeyBERT()
sentBERT_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_jsonl(file_path):
     """  Loads data from a JSONL file with the correct multi-document format.
          Each line represents one sample to multi-doc summarization
     """
     documents_list = []
     with open(file_path, 'r', encoding='utf-8') as f:
          for line in f:
               try:
                    data = json.loads(line)
                    documents_list.append(data['text']) # Each line is a list of documents
               except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
     return documents_list


def split_sentences(documents_list):
     """Splits sentences in each document within the list."""
     processed_documents_list = []
     for documents in documents_list:
          processed_documents = []
          for document in documents:
               sentences = []
               doc = nlp_sm(document[0])
               for sent in doc.sents:
                    sentences.append(sent.text)
               processed_documents.append(sentences)
          processed_documents_list.append(processed_documents)

     return processed_documents_list


def extract_keywords(documents_list, words_per_100=1, min_keywords=2, max_keywords=15):
     """extract key word from each document, default 10 words

     Args:
          documents_list (_type_): original documents sample list. No need to split document

     Returns:
          _type_: list, keywords of each document
     """
     
     keywords_list = []
     for documents in documents_list:
          kw_per_doc = []
          for doc in documents:
               ## dynamic top_n
               top_n = max(min_keywords, min(max_keywords, (len(doc) // 100) * words_per_100))
               
               keywords = kw_model.extract_keywords(
                    doc,
                    top_n=top_n,
                    stop_words='english',
                    use_mmr=True,
                    diversity=0.6,
                    keyphrase_ngram_range=(1, 1)
               )
               kw_per_doc.append(keywords)

          keywords_list.append(kw_per_doc)
     return keywords_list


def coref_resolve(documents_list):
     """_summary_ coreference resolve

     Args:
          documents_list (_type_): documents with sentences
          sent_sentid_map (_type_): sentence to sentence id dictionary, key: (training_idx, doc_idx, sent), value: node id

     Returns:
          _type_: coref_cluster for each document. form as orginal dataset. cluster element: (training_idx, doc_idx, sent), to find node id.
     """
     coref_docs_list= []
     for training_id, docs in enumerate(documents_list):
          coref_docs = []
          for doc_id, document in enumerate(docs):
               doc = nlp_coref(document[0])
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


def define_node_edge(documents_list, edge_similarity_threshold = 0.6):
     """_summary_ index node, including word and sentence; define all types of edges

     Args:
          documents_list (_type_): original docs

     Returns:
          _type_:  word_node_map, sent_node_map, edge list, sentid_nodeid_map
     """
     docs_sents_list = split_sentences(documents_list)
     docs_kws_scores_list = extract_keywords(documents_list)
     docs_corefs_list = coref_resolve(documents_list)
     
     edge_data_list = []
     word_node_list = []
     sent_node_list = []
     sentId_nodeId_list = []
     for training_idx, (docs_sentences, docs_kws_scores, docs_corefs) in enumerate(zip(docs_sents_list, docs_kws_scores_list, docs_corefs_list)):
          node_index = 0
          
          # sentence node index
          sent_nodeId_map = {} ## for sentence related edges
          sentId_nodeId_map = {}
          for doc_idx, sents in enumerate(docs_sentences):
               for sent_id, sent in enumerate(sents):
                    sent_nodeId_map[(training_idx, doc_idx, sent)] = node_index
                    sentId_nodeId_map[(training_idx, doc_idx, sent_id)] = node_index
                    node_index += 1
          num_sentences = node_index

          # word node index
          word_nodeId_map = {} ## for word-sentence edge
          word_index = num_sentences
          for doc_idx, doc_kws_scs in enumerate(docs_kws_scores):
               for keyword, score in doc_kws_scs:
                    if keyword not in word_nodeId_map:
                         word_nodeId_map[keyword] = word_index
                         word_index += 1
          
          edge_data = defaultdict(list)
          def add_edge(node1_idx, node2_idx, edge_type, weight=1):
               if torch.is_tensor(weight):
                    weight = weight.detach().item()
               edge_data[(node1_idx, node2_idx)].append({'type': edge_type, 'weight': weight})
               
          ## 1. word-sentence
          for doc_idx, sents in enumerate(docs_sentences):
               for sent_id, sent in enumerate(sents):
                    ## check wether contain keywords
                    sent_low = sent.lower()
                    for word, score in docs_kws_scores[doc_idx]:
                         if word.lower() in sent_low:
                              word_node = word_nodeId_map[word]
                              sent_node = sentId_nodeId_map[(training_idx, doc_idx, sent_id)]
                              add_edge(word_node, sent_node, "word_sent", weight=score)
                              
          ## 2. pronoun-antecedent
          for doc_idx, doc_corefs in enumerate(docs_corefs): ## as [(training_id, doc_id, sent_text)...]
               for corf_cluster in doc_corefs:
                    antecedent = sent_nodeId_map[corf_cluster[0]]
                    for i in range(1, len(corf_cluster)):
                         ## edge on each coref. the first one is the resolve
                         add_edge(sent_nodeId_map[corf_cluster[i]], antecedent, "pronoun_antecedent")
          
          ## 3. similarity
          for doc_idx, sents in enumerate(docs_sentences):
               sent_embeddings = sentBERT_model.encode(sents)
               similarities_matrix = sentBERT_model.similarity(sent_embeddings, sent_embeddings)
               
               n = len(sent_embeddings)
               for i in range(n):
                    for j in range(n):
                         if j < i and similarities_matrix[i][j] >= edge_similarity_threshold:
                              node_i = sentId_nodeId_map[(training_idx, doc_idx, i)]
                              node_j = sentId_nodeId_map[(training_idx, doc_idx, j)]
                              
                              add_edge(node_i, node_j, "similarity", weight=similarities_matrix[i][j])
          
          
          edge_data_list.append(edge_data)
          word_node_list.append(word_nodeId_map)
          sent_node_list.append(sent_nodeId_map)
          sentId_nodeId_list.append(sentId_nodeId_map)
     
     return word_node_list, sent_node_list, edge_data_list, sentId_nodeId_list