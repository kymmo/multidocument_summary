import os
import torch
import dill
from pathlib import Path
from datetime import datetime
from enum import Enum

parent_path = "/content/drive/MyDrive/checkpoints"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelCheckpointManager:
     """Training checkpoints manager, auto recover."""
     def __init__(self,
                    checkpoint_dir=os.path.join(parent_path, 'models'),
                    max_keep=3,
                    stage_name="stage1"):
          self.checkpoint_dir = checkpoint_dir
          self.max_keep = max_keep
          self.stage_name = stage_name
          os.makedirs(self.checkpoint_dir, exist_ok=True)

     def _get_filepath(self, emergency=False, epoch=None):
          timestamp = datetime.now().strftime("%m%d-%H%M")
          if emergency:
               return os.path.join(self.checkpoint_dir,
                                   f"{self.stage_name}_emergency_{timestamp}.pth")
          return os.path.join(self.checkpoint_dir,
                              f"{self.stage_name}_epoch{epoch}.pth")

     def save(self, epoch, models, optimizers, schedulers, scaler=None, **kwargs):
          checkpoint = {
               'epoch': epoch,
               'timestamp': datetime.now().isoformat(),
               'stage': self.stage_name,
               'scaler': scaler.state_dict() if scaler else None
          }

          for name, obj in models.items():
               checkpoint[f"{name}_state"] = obj.state_dict()
          for name, obj in optimizers.items():
               checkpoint[f"{name}_state"] = obj.state_dict()
          for name, obj in schedulers.items():
               checkpoint[f"{name}_state"] = obj.state_dict()
          
          checkpoint.update(kwargs)

          filepath = self._get_filepath(epoch=epoch)
          torch.save(checkpoint, filepath)
          self._clean_old_checkpoints()
          return filepath

     def load(self, device=device):
          checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                         if f.startswith(self.stage_name) and "emergency" not in f]
          if not checkpoints:
               return None
          
          latest = sorted(checkpoints)[-1]
          filepath = os.path.join(self.checkpoint_dir, latest)
          checkpoint = torch.load(filepath, map_location=device)
          
          ## version check
          required_keys = ['stage', 'epoch', 'timestamp']
          if not all(k in checkpoint for k in required_keys):
               raise ValueError("Invalid checkpoint format")
          
          return checkpoint

     def _clean_old_checkpoints(self):
          """only keep max_keep checkpoints"""
          checkpoints = [f for f in os.listdir(self.checkpoint_dir)
                         if f.startswith(self.stage_name) and "emergency" not in f]
          for f in checkpoints[:-self.max_keep]:
               os.remove(os.path.join(self.checkpoint_dir, f))
               

class DataCheckpointManager:
     class StepKey(Enum):
          PREDEFINE = "define_node"
          GRAPH = "create_graph"
          EMBED = "embed_graph"
          FINAL = "final_data"
          
     def __init__(self, 
                    save_dir = os.path.join(parent_path, 'data')):
          self.save_dir = Path(save_dir)
          self.step_files = {
               self.StepKey.PREDEFINE.value: 'step1_define_node.pkl',
               self.StepKey.GRAPH.value: 'step2_create_graph.pkl',
               self.StepKey.EMBED.value: 'step3_embed_graph.pkl',
               self.StepKey.FINAL.value: 'final_data.pkl'
          }
          self.save_dir.mkdir(parents=True, exist_ok=True)
     
     def get_latest_step(self):
          completed_steps = []
          for step, fname in self.step_files.items():
               if (self.save_dir / fname).exists():
                    completed_steps.append(step)
          
          ## order check
          for i in range(4):
               if i < len(completed_steps):
                    if (i == 0 and completed_steps[i] != self.StepKey.PREDEFINE.value) or \
                    (i == 1 and completed_steps[i] != self.StepKey.GRAPH.value) or \
                    (i == 2 and completed_steps[i] != self.StepKey.EMBED.value) or \
                    (i == 3 and completed_steps[i] != self.StepKey.FINAL.value):
                         print("Error! Preprocess data has wrong order!")
                         return None
               
          return completed_steps[-1] if completed_steps else None
     
     def save_step(self, step_name, data):
          fpath = self.save_dir / self.step_files[step_name]
          with open(fpath, 'wb') as f:
               dill.dump({
                    'step': step_name,
                    'timestamp': datetime.now().strftime("%m%d-%H%M"),
                    'data': data
               }, f)
          print(f"Current process [{step_name}] data has been saved to path {fpath}")
     
     def load_step(self, step_name):
          """load specific step"""
          fpath = self.save_dir / self.step_files[step_name]
          try:
               with open(fpath, 'rb') as f:
                    data = dill.load(f)
                    
                    return data['data']
          except Exception as e:
               print(f"Processed data [{step_name}] load fail. Error: {str(e)}")
               return None