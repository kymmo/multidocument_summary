from torch_geometric.data import HeteroData
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from torch_geometric.data import Batch

from utils.graph_utils import get_embed_graph, get_embed_graph_node_map


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
                         txt = het_graph[node_type].text ### TODO: str list to copy to shared momery
                         shared_graph[node_type].x = x
                         shared_graph[node_type].text = txt
                         
                    for edge_type in het_graph.edge_types:
                         edge_index = het_graph[edge_type].edge_index.clone().share_memory_()
                         edge_attr = het_graph[edge_type].edge_attr.clone().share_memory_()
                         shared_graph[edge_type].edge_index = edge_index
                         shared_graph[edge_type].edge_attr = edge_attr
                    
                    self.data.append(shared_graph)
               
               self._loaded.value = True  # marked as loaded
               print(f"Data has been loaded into memory!")

     def __len__(self):
          if not self._loaded.value:
               self._load_all()  # late loader
               
          return len(self.data)

     def __getitem__(self, idx):
          if not self._loaded.value:
               self._load_all()
               
          return self.data[idx].cpu()
     
def custom_collate_fn(batch):
     graphs, node_maps, summary_list = zip(*batch)
     # batched_graph = Batch.from_data_list(graphs)
     
     batched_maps = []
     batch_summary = []
     batched_graph = []
     for graph, node_map, summary in zip(graphs, node_maps, summary_list):
          batched_maps.append(node_map)
          batch_summary.append(summary) ## string list
          batched_graph.append(graph)
     
     return batched_graph, batched_maps, batch_summary