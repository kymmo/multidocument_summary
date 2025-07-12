from torch_geometric.data import HeteroData
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from torch_geometric.data import Batch
import torch

from utils.graph_utils import get_embed_graph, get_embed_graph_node_map


class EvalDataset(Dataset):
     def __init__(self, file_path, dataset_type, sent_similarity):
          self.file_path = file_path
          self.dataset_type = dataset_type
          self.sent_similarity = sent_similarity
          self.data, self.node_map, self.summary_list = self._load_data(file_path)

     def _load_data(self, file_path):
          embedded_graphs, node_maps, summary_list = \
          get_embed_graph_node_map(file_path=file_path, dataset_type=self.dataset_type, sent_similarity=self.sent_similarity) ## node_maps: sent_node_id-> sent_text
          
          return embedded_graphs, node_maps, summary_list

     def __len__(self):
          return len(self.data)

     def __getitem__(self, idx):
          return self.data[idx], self.node_map[idx], self.summary_list[idx]

class OptimizedDataset(Dataset):
     def __init__(self, file_path, dataset_type, sent_similarity):
          self.file_path = file_path
          self.data = None
          self._lock = mp.Lock()
          self._loaded = mp.Value('b', False) ## multi-process shared
          self.dataset_type = dataset_type
          self.sent_similarity = sent_similarity

     def _load_all(self):
          with self._lock:
               if self._loaded.value:
                    return
               
               raw_data = get_embed_graph(self.file_path, dataset_type=self.dataset_type, sent_similarity=self.sent_similarity)
               
               self.data = raw_data
               self._loaded.value = True  # marked as loaded
          
     def __len__(self):
          if not self._loaded.value:
               self._load_all()  # late loader
               
          return len(self.data)

     def __getitem__(self, idx):
          if not self._loaded.value:
               self._load_all()
          
          if self.data is None or self.data[idx] is None:
               raise ValueError("Data not loaded yet. Current value is None.")
          
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

class JointTrainingDataset(Dataset):
     """
     Loads the raw graph object (with initial embeddings and raw sentence text)
     and the corresponding summary string for joint training.
     """
     def __init__(self, file_path, dataset_type, sent_similarity):
          self.dataset_type = dataset_type
          self.sent_similarity = sent_similarity
          self.graphs, self.summaries = self._load_data(file_path)

     def _load_data(self, file_path):
          embedded_graphs, node_maps, summary_list = \
          get_embed_graph_node_map(file_path=file_path, dataset_type=self.dataset_type, sent_similarity=self.sent_similarity) ## node_maps: sent_node_id-> sent_text
          
          return embedded_graphs, summary_list

     def __len__(self):
          return len(self.graphs)

     def __getitem__(self, idx):
          return self.graphs[idx], self.summaries[idx]

def joint_collate_fn(batch):
     graphs, summaries = zip(*batch)
     graph_list = list(graphs)
     batched_graph = Batch.from_data_list(graph_list)
     
     text_list = [] ## sample text string list, one string for one sample
     for graph in graph_list:
          sent_texts = graph['sentence'].text
          
          sample_text = " ".join(sent_texts)
          text_list.append(sample_text)
     
     
     return {
          'batched_graph': batched_graph,
          'label_summaries': list(summaries),
          'graph_list': graph_list,
          'sample_text_list': text_list,
     }