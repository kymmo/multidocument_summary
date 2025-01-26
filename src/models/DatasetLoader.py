from torch.utils.data import Dataset

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
          return len(self.data)

     def __getitem__(self, idx):
          return self.data[idx]
     
     
class EvalDataset(Dataset):
     def __init__(self, file_path):
          self.file_path = file_path
          self.data, self.node_map = self._load_data(file_path)

     def _load_data(self, file_path):
          embedded_graphs, node_maps = get_embed_graph_node_map(file_path) ## node_maps: sent_node_id-> sent_text
          return embedded_graphs, node_maps

     def __len__(self):
          return len(self.data)

     def __getitem__(self, idx):
          return self.data[idx], self.node_map[idx]