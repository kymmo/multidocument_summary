import os
import torch
import shutil
import traceback
import json
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast

from models.CustomT5 import CustomT5
from models.JoinModel import JointOrchestrator, JointOrchestratorwithPrefix
from utils.model_utils import clean_memory


class ModelFileManager:
     def __init__(self):
          self.save_parent_dir = os.path.join("/", "content", "drive", "MyDrive", "saved_models")
          self.gnn_model_dir = os.path.join(self.save_parent_dir, "gnn_trained_weights.pt")
          self.t5_model_dir = os.path.join(self.save_parent_dir, "fine_tuned_t5")
          self.joint_model_dir = os.path.join(self.save_parent_dir, "join_orchestrator_model")
     
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
               print(f"Found exist {self.t5_model_dir} folder. Deleting it.")
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
     
     def save_join_model(self, model: JointOrchestratorwithPrefix):
          os.makedirs(self.joint_model_dir, exist_ok=True)

          config_to_save = model.config
          config_path = os.path.join(self.joint_model_dir, "config.json")
          with open(config_path, 'w') as f:
               json.dump(config_to_save, f, indent=4)
               
          model.tokenizer.save_pretrained(self.joint_model_dir)

          trainable_state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
          weights_path = os.path.join(self.joint_model_dir, "pytorch_model.bin")
          
          torch.save(trainable_state_dict, weights_path)
          print(f"Join-Model saved successfully to {self.joint_model_dir}.")

     def load_join_model(self):
          config_path = os.path.join(self.joint_model_dir, "config.json")
          if not os.path.exists(config_path):
               raise FileNotFoundError(f"Config file not found at {config_path}")
          with open(config_path, 'r') as f:
               config = json.load(f)

          tokenizer = T5Tokenizer.from_pretrained(self.joint_model_dir)

          model = JointOrchestratorwithPrefix(
               gnn_config=config["gnn_config"],
               t5_model_name=config["t5_model_name"],
               prefix_length=config["prefix_length"],
               t5_tokenizer=tokenizer
          )

          weights_path = os.path.join(self.joint_model_dir, "pytorch_model.bin")
          if not os.path.exists(weights_path):
               raise FileNotFoundError(f"Weights file not found at {weights_path}")
               
          loaded_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
          model.load_state_dict(loaded_state_dict, strict=False)
          
          print("Model loaded successfully.")
          
          return model
     
# Singleton instance
model_fm = ModelFileManager()