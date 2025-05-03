import os
import torch
import shutil
import traceback

from models.CustomT5 import CustomT5


class ModelFileManager:
     def __init__(self):
          self.save_parent_dir = os.path.join("content", "drive", "MyDrive", "saved_models")
          self.gnn_model_dir = os.path.join(self.save_parent_dir, "gnn_trained_weights.pt")
          self.t5_model_dir = os.path.join(self.save_parent_dir, "fine_tuned_t5")
          
     def save_gnn(self, gnn_model):
          if not os.path.exists(self.save_parent_dir):
               os.makedirs(self.save_parent_dir)
               
          if os.path.exists(self.gnn_model_dir):
               print(f"Found exist {self.gnn_model_dir} file. Deleting it.")
               os.remove(self.gnn_model_dir)
          
          try:
               torch.save(gnn_model, self.gnn_model_dir)
          except Exception as e:
               print(f"[Error] while saving GNN model: {e}")
               traceback.print_exc()
          
     def load_gnn(self):
          if os.path.exists(self.gnn_model_dir):
               gnn_model = torch.load(self.gnn_model_dir)
               return gnn_model
          else:
               raise FileNotFoundError(f"GNN model not found at {self.gnn_model_dir}")
          
     def save_t5(self, t5_model):
          if not os.path.exists(self.save_parent_dir):
               os.makedirs(self.save_parent_dir)
               
          if os.path.exists(self.t5_model_dir):
               print(f"Found exist {self.t5_model_dir} file. Deleting it.")
               shutil.rmtree(self.t5_model_dir)
          
          try:
               t5_model.save_pretrained(self.t5_model_dir)
          except Exception as e:
               print(f"[Error] while saving T5 model: {e}")
               traceback.print_exc()
          
     def load_t5(self):
          if os.path.exists(self.t5_model_dir):
               fine_tuned_t5 = CustomT5.from_pretrained(self.t5_model_dir, low_cpu_mem_usage=True)
               return fine_tuned_t5
          else:
               raise FileNotFoundError(f"T5 model not found at {self.t5_model_dir}")
          
# Singleton instance
model_fm = ModelFileManager()