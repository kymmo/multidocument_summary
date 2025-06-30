import os
import torch
import shutil
import traceback
import json
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast

from models.CustomT5 import CustomT5
from models.JointOrchestrator import JointOrchestrator
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
     
     def save_joint_model(self, model: JointOrchestrator, gnn_config: dict, t5_tokenizer: T5Tokenizer):
          if not os.path.exists(self.save_parent_dir):
               os.makedirs(self.save_parent_dir)
               
          if os.path.exists(self.joint_model_dir):
               print(f"Found exist {self.joint_model_dir} folder. Deleting it.")
               try:
                    shutil.rmtree(self.joint_model_dir)
               except OSError as e:
                    print(f"Error removing directory {self.joint_model_dir}: {e.strerror}")
                    return
               
          os.makedirs(self.joint_model_dir, exist_ok=True)
          clean_memory()
          
          try:
               # 1. Save the CustomT5 component using Hugging Face's powerful method.
               # This saves its config and weights.
               t5_path = os.path.join(self.joint_model_dir, "custom_t5")
               model.custom_t5.save_pretrained(t5_path)
               
               # 2. Save the GNN's state dictionary
               torch.save(model.gnn.state_dict(), os.path.join(self.joint_model_dir, "gnn_weights.pth"))
               
               # 3. Save the GNN's configuration dictionary as a JSON file.
               # This is KEY to not needing to pre-define it on load.
               with open(os.path.join(self.joint_model_dir, "gnn_config.json"), 'w') as f:
                    json.dump(gnn_config, f, indent=4)

               # 4. Save the Text Encoder's state dictionary
               torch.save(model.text_encoder.model.state_dict(), os.path.join(self.joint_model_dir, "text_encoder_weights.pth"))
               
               # 5. Save the T5 Tokenizer
               t5_tokenizer.save_pretrained(self.joint_model_dir)

               print("Joint model components saved successfully.")
          except Exception as e:
               print(f"[Error] during hybrid save of Joint Orchestrator model: {e}")
               traceback.print_exc()

     def load_joint_model(self):
          """
          Loads the entire JointOrchestrator model from the saved directory
          without needing to pre-define configurations.
          """
          if not os.path.exists(self.joint_model_dir):
               raise FileNotFoundError(f"Joint model directory not found at {self.joint_model_dir}")

          # 1. Load the T5 Tokenizer
          t5_tokenizer = T5TokenizerFast.from_pretrained(self.joint_model_dir)

          # 2. Load the GNN config from its JSON file
          with open(os.path.join(self.joint_model_dir, "gnn_config.json"), 'r') as f:
               gnn_config = json.load(f)
               
          # 3. Load the T5 config from the saved CustomT5 directory
          custom_t5_path = os.path.join(self.joint_model_dir, "custom_t5")
          t5_config = T5Config.from_pretrained(custom_t5_path)
          
          # 4. Instantiate the base T5 model for the Text Encoder
          # We use the name from the tokenizer files.
          text_encoder_model = T5ForConditionalGeneration(t5_config)
          text_encoder_weights_path = os.path.join(self.joint_model_dir, "text_encoder_weights.pth")
          text_encoder_model.load_state_dict(
               torch.load(text_encoder_weights_path, map_location="cpu")
          )
          
          # 6. Now, instantiate the full JointOrchestrator "shell"
          # We have all the configs and models we need to do this.
          model = JointOrchestrator(
               gnn_config=gnn_config,
               t5_config=t5_config,
               text_encoder_model=text_encoder_model,
               t5_tokenizer=t5_tokenizer
          )
          
          # 7. Load the saved state dicts for the GNN and the CustomT5 part
          model.gnn.load_state_dict(torch.load(os.path.join(self.joint_model_dir, "gnn_weights.pth")))
          
          # The CustomT5's state is more complex because it's a full HF model.
          # We load it using from_pretrained.
          model.custom_t5 = CustomT5.from_pretrained(custom_t5_path)
          
          print("Joint model fully loaded and assembled successfully.")
          return model
     
# Singleton instance
model_fm = ModelFileManager()