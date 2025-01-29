from utils.data_preprocess_utils import define_node_edge, split_sentences, coref_resolve, extract_keywords,load_jsonl
from utils.graph_utils import create_graph
import networkx as nx

if __name__ == '__main__':
     
     documents_list = [
          [["This is the first document. It has two sentences."], ["Document 2 text."]],
          [["Document 3 text. I like eda."], ["Here's the second document. It also contains two sentences."]]
     ]
     # result1 = split_sentences(documents_list)
     # result2 = coref_resolve(documents_list)
     # result3 = extract_keywords(documents_list)
     
     # print("split_sent: ", result1)
     # print("coref_res: ", result2)
     # print("keywords: ", result3)
     
     word_node_list, sent_node_list, edge_data_list, sentId_nodeId_list = define_node_edge(documents_list)
     print("word_node_list: ", word_node_list)
     print("sent_node_list: ", sent_node_list)
     print("edge_data_list: ", edge_data_list)
     print("sentId_nodeId_list: ", sentId_nodeId_list)
     
     graphs = create_graph(word_node_list, sent_node_list, edge_data_list)
     for id, g in enumerate(graphs):
          print(f"graph {id}: ")
          print("Nodes:")
          print(g.nodes())
          print("\nEdges:")
          print(g.edges(data=True, keys=True))