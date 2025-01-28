import torch
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset
import torch.multiprocessing as mp

from utils.graph_utils import get_embed_graph, get_embed_graph_node_map

class SummaryDataset(Dataset):
     def __init__(self, file_path):
          self.file_path = file_path
          self.data = self._load_data(file_path)

     def _load_data(self, file_path):
          ## should load a list of embeded hetgraph
          embedded_graphs = get_embed_graph(file_path)
          return embedded_graphs

     def __len__(self):
          print("len ", len(self.data))
          return len(self.data)

     def __getitem__(self, idx):
          ## return HeteroData
          print(f"get_item: id {idx}, data {self.data[idx]}")
          return self.data[idx]
     
     
class EvalDataset(Dataset):
     def __init__(self, file_path):
          self.file_path = file_path
          self.data, self.node_map, self.summary_list = self._load_data(file_path)

     def _load_data(self, file_path):
          embedded_graphs, node_maps, summary_list = get_embed_graph_node_map(file_path) ## node_maps: sent_node_id-> sent_text
          
          return embedded_graphs, node_maps, summary_list

     def __len__(self):
          return len(self.data)

     def __getitem__(self, idx):
          return self.data[idx], self.node_map[idx], self.summary_list[idx]
     

class OptimizedDataset(Dataset):
     def __init__(self, file_path):
          self.file_path = file_path
          self.data = None
          self._lock = mp.Lock()
          self._loaded = mp.Value('b', False) ## multi-process shared

     def _load_all(self):
          with self._lock:
               if self._loaded.value:
                    return
               
               print("Loading data into shared memory...")
               raw_data = get_embed_graph(self.file_path)
               
               self.data = []
               for het_graph in raw_data:
                    shared_graph = HeteroData()
                    
                    # conver the embedding to shared memory space
                    for node_type in het_graph.node_types:
                         x = het_graph[node_type].x.clone().share_memory_()
                         shared_graph[node_type].x = x
                    
                    for edge_type in het_graph.edge_types:
                         edge_index = het_graph[edge_type].edge_index.clone().share_memory_()
                         edge_attr = het_graph[edge_type].edge_attr.clone().share_memory_()
                         shared_graph[edge_type].edge_index = edge_index
                         shared_graph[edge_type].edge_attr = edge_attr
                    
                    self.data.append(shared_graph)
               
               self._loaded.value = True  # marked as loaded

     def __len__(self):
          if not self._loaded.value:
               self._load_all()  # late loader
          print("len ", len(self.data))
          return len(self.data)

     def __getitem__(self, idx):
          if not self._loaded.value:
               self._load_all()
               
          print(f"get_item: id {idx}, data {self.data[idx]}")
          return self.data[idx]