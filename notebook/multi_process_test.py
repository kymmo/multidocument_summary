from utils.data_preprocess_utils import define_node_edge, split_sentences, coref_resolve, extract_keywords,load_jsonl, split_sentences2, baseline_split_sentences
from utils.graph_utils import create_graph
import networkx as nx
import time

if __name__ == '__main__':
     
     documents_list = [
          [["This is the first document. It has two sentences."], ["Document 2 text."], ["here is the third one. sample 1. ahhhhh!"]],
          [["sample 2 document 1 text. I like eda."], ["Here's the second document. It also contains two sentences."]]
     ]

     dataset, summary_list = load_jsonl(".\data\\train_pro.jsonl")
     start1_1 = time.time()
     result1_1 = split_sentences2(dataset)
     end1_1 = time.time()
     print(f"split_sent: improved:. time cost:  {end1_1 - start1_1:.4f} s.")
     
     # start1 = time.time()
     # result1 = split_sentences(dataset)
     # end1 = time.time()
     # print(f"split_sent: original:. time cost:  {end1 - start1:.4f} s.")
     
     start0 = time.time()
     result0 = baseline_split_sentences(dataset)
     end0 = time.time()
     print(f"split_sent: single:. time cost:  {end0 - start0:.4f} s.")
     
     if not start1_1 == result0:
          print("列表元素不相等！")
          print("improved: ", start1_1[1])
          print("original: ", result0[1])
          
     # result2 = coref_resolve(documents_list)
     # result3 = extract_keywords(documents_list)
     
     # print("coref_res: ", result2)
     # print("keywords: ", result3)
     
     # word_node_list, sent_node_list, edge_data_list, sentId_nodeId_list = define_node_edge(documents_list)
     # print("word_node_list: ", word_node_list)
     # print("sent_node_list: ", sent_node_list)
     # print("edge_data_list: ", edge_data_list)
     # print("sentId_nodeId_list: ", sentId_nodeId_list)
     
     # graphs = create_graph(word_node_list, sent_node_list, edge_data_list)
     # for id, g in enumerate(graphs):
     #      print(f"graph {id}: ")
     #      print("Nodes:")
     #      print(g.nodes())
     #      print("\nEdges:")
     #      print(g.edges(data=True, keys=True))