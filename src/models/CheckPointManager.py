import os
import torch
import dill
from pathlib import Path
from datetime import datetime
from enum import Enum

parent_path = os.path.join("/","content", "drive", "MyDrive", "checkpoints")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelCheckpointManager:
     """Training checkpoints manager, auto recover."""
     def __init__(self,
                    checkpoint_dir=os.path.join(parent_path, 'models'),
                    max_keep=4,
                    stage_name="stage1"):
          self.checkpoint_dir = checkpoint_dir
          self.max_keep = max_keep
          self.stage_name = stage_name
          os.makedirs(self.checkpoint_dir, exist_ok=True)

     def _get_filepath(self, epoch=None, emergency=False, is_best=False):
          timestamp = datetime.now().strftime("%m%d-%H%M")
          if is_best:
               return os.path.join(self.checkpoint_dir, f"{self.stage_name}_best.pth")
          
          if emergency:
               return os.path.join(self.checkpoint_dir,
                                   f"{self.stage_name}_emergency_{timestamp}.pth")
          if epoch is not None:
               return os.path.join(self.checkpoint_dir,
                                   f"{self.stage_name}_epoch{epoch}.pth")
          # a fallback if no identifier provided
          return os.path.join(self.checkpoint_dir, f"{self.stage_name}_checkpoint_{timestamp}.pth")
     
     def save(self, models, optimizers, schedulers, epoch=None, scaler=None, is_best=False, **kwargs):
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
          checkpoint['is_best'] = is_best
          
          checkpoint.update(kwargs)

          filepath = self._get_filepath(epoch=epoch, is_best=is_best)
          torch.save(checkpoint, filepath)
          
          if not is_best:
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
          
          ################test
          print(f"original checkpoints: {checkpoints}")
          print(f"latest: {latest}")
          print(f"file path: {filepath}")
          print(f"checkpoint epoch: {checkpoint['epoch']}")
          ###########################
          ## version check
          required_keys = ['stage', 'epoch', 'timestamp']
          if not all(k in checkpoint for k in required_keys):
               raise ValueError("Invalid checkpoint format")
          
          return checkpoint
               
     def _clean_old_checkpoints(self):
          checkpoints = [f for f in os.listdir(self.checkpoint_dir)
                         if f.startswith(f"{self.stage_name}_epoch") and f.endswith(".pth")] # Only match epoch files
          checkpoints.sort(key=lambda x: int(x.split('epoch')[-1].split('.pth')[0]))

          if len(checkpoints) > self.max_keep:
               for f in checkpoints[:-self.max_keep]:
                    file_to_remove = os.path.join(self.checkpoint_dir, f)
                    print(f"[Checkpoint Manager] Removing old checkpoint: {file_to_remove}")
                    os.remove(file_to_remove)
     
     def load_best(self, device=device):
          best_filepath = os.path.join(self.checkpoint_dir, f"{self.stage_name}_best.pth")
          if os.path.exists(best_filepath):
               checkpoint = torch.load(best_filepath, map_location=device)

               return checkpoint
          else:
               print("[Checkpoint Manager] Best checkpoint not found.")
               return None
     

class DataType(Enum):
     TRAIN = 'train'
     VALIDATION = 'val'
     TEST = 'test'
          
class DataCheckpointManager:
     class StepKey(Enum):
          PREDEFINE = "define_node"
          GRAPH = "create_graph"
          EMBED = "embed_graph"
          FINAL = "final_graph"

     def __init__(self, save_dir = os.path.join(parent_path, 'data')):
          self.save_dir = Path(save_dir)

          self._step_base_filenames = {
               self.StepKey.PREDEFINE.value: f'step1_{self.StepKey.PREDEFINE.value}.pkl',
               self.StepKey.GRAPH.value: f'step2_{self.StepKey.GRAPH.value}.pkl',
               self.StepKey.EMBED.value: f'step3_{self.StepKey.EMBED.value}.pkl',
               self.StepKey.FINAL.value: f'step4_{self.StepKey.FINAL.value}.pkl'
          }
          self.save_dir.mkdir(parents=True, exist_ok=True)

          self._step_order = [
               self.StepKey.PREDEFINE.value, 
               self.StepKey.GRAPH.value, 
               self.StepKey.EMBED.value, 
               self.StepKey.FINAL.value
          ]

     def _get_filepath(self, step_name: str, dataset_type: str) -> Path:
          """Constructs the full filepath for a given step and dataset type."""
          if step_name not in self._step_base_filenames:
               raise ValueError(f"Unknown step name: {step_name}")

          valid_types = [dt.value for dt in DataType]
          if dataset_type not in valid_types:
               raise ValueError(f"Invalid dataset_type: {dataset_type}. Use one of {valid_types}")

          base_filename = self._step_base_filenames[step_name]

          filename = f"{dataset_type}_{base_filename}"
          return self.save_dir / filename

     def get_latest_step(self, dataset_type: str):
          """Gets the latest completed processing step for a specific dataset type."""
          latest_found_step = None
          for step_name in self._step_order:
               fpath = self._get_filepath(step_name, dataset_type)
               if fpath.exists():
                    latest_found_step = step_name
               else:
                    # If a step is missing, the chain is broken for subsequent steps
                    break

          return latest_found_step

     def save_step(self, step_name: str, dataset_type: str, data):
          """Saves the data for a specific step and dataset type."""
          fpath = self._get_filepath(step_name, dataset_type)
          try:
               with open(fpath, 'wb') as f:
                    dill.dump({
                         'step': step_name,
                         'dataset_type': dataset_type,
                         'timestamp': datetime.now().isoformat(), # Use ISO format
                         'data': data
                    }, f)
               print(f"[Data Checkpoint] {dataset_type} data for step [{step_name}] saved to: {fpath}")
          except Exception as e:
               raise e

     def load_step(self, step_name: str, dataset_type: str):
          """Loads the data for a specific step and dataset type."""
          fpath = self._get_filepath(step_name, dataset_type)
          if not fpath.exists():
               print(f"[Data Checkpoint] Checkpoint file not found for step [{step_name}], type [{dataset_type}] at: {fpath}")
               return None
          
          try:
               with open(fpath, 'rb') as f:
                    loaded_data = dill.load(f)
                    if 'data' not in loaded_data or 'step' not in loaded_data or 'dataset_type' not in loaded_data:
                         print(f"Warning: Checkpoint file {fpath} has unexpected format.")
                         return None
                    
                    if loaded_data.get('step') != step_name or loaded_data.get('dataset_type') != dataset_type:
                         print(f"Warning: Metadata mismatch in checkpoint file {fpath}. Expected step='{step_name}', type='{dataset_type}', but got step='{loaded_data.get('step')}', type='{loaded_data.get('dataset_type')}'.")
                         return None
                    
                    print(f"[Data Checkpoint] {dataset_type} data for step [{step_name}] loaded.")
                    return loaded_data['data']
          except (dill.UnpicklingError, EOFError, FileNotFoundError) as e:
               print(f"Error loading data for step [{step_name}], type [{dataset_type}] from {fpath}: {e}")
               return None
          except Exception as e:
               print(f"An unexpected error occurred loading data for step [{step_name}], type [{dataset_type}] from {fpath}: {e}")
               return None