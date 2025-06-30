
class EarlyStopper:
     def __init__(self, patience=5, min_delta=0, checkpoint_manager=None):
          self.patience = patience
          self.min_delta = min_delta
          self.counter = 0
          self.best_loss = float('inf')
          self.early_stop_triggered = False
          self.stopped_epoch = None
          
          if checkpoint_manager is None:
               raise ValueError("EarlyStopper requires a ModelCheckpointManager instance.")
          
          self.checkpoint_manager = checkpoint_manager

     def __call__(self, val_loss, epoch, models, optimizers, schedulers, scaler, global_step = None,
                    train_losses_history=None, val_losses_history=None):
          if val_loss < self.best_loss - self.min_delta:
               self.best_loss = val_loss
               self.counter = 0
               self.stopped_epoch = None
               self.early_stop_triggered = False
               
               best_state_to_save = self.get_state()

               print(f"[EarlyStop] Validation loss improved to {val_loss:.4f}. Saving best model...")
               save_kwargs = {
                    'epoch': epoch,
                    'models': models,
                    'optimizers': optimizers,
                    'schedulers': schedulers,
                    'scaler': scaler,
                    'is_best': True,
                    'early_stopper_state': best_state_to_save
               }
               if global_step is not None:
                    save_kwargs['global_step'] = global_step

               if train_losses_history is not None: save_kwargs['train_losses'] = train_losses_history
               if val_losses_history is not None: save_kwargs['val_losses'] = val_losses_history
               
               best_filepath = self.checkpoint_manager.save(**save_kwargs)
               print(f"[EarlyStop] Best model saved to {best_filepath}.")
          else:
               self.counter += 1
               print(f"[EarlyStop] No improvement for {self.counter}/{self.patience} epochs.")
               if self.counter >= self.patience:
                    print("[EarlyStop] Stopping training.")
                    self.early_stop_triggered = True
                    self.stopped_epoch = epoch
          
          return self.early_stop_triggered
     
     def get_state(self):
          return {
               'best_loss': self.best_loss,
               'counter': self.counter,
               'stopped_epoch': self.stopped_epoch,
               'early_stop_triggered': self.early_stop_triggered
          }

     def load_state(self, state_dict):
          """Loads the best state into the early stopper."""

          self.best_loss = state_dict.get('best_loss', float('inf'))
          self.counter = state_dict.get('counter', 0)
          self.stopped_epoch = state_dict.get('stopped_epoch', None)
          self.early_stop_triggered = state_dict.get('early_stop_triggered', False)
          print(f"[EarlyStop] Loaded state: {self.get_state()}")