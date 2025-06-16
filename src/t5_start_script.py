import argparse
import sys
import traceback

def run():
     """
     Main function to parse arguments and run the T5 training/evaluation.
     
     function signature: 
     def model_train_eval(dataset_path, learning_rate = 0.001,num_epochs = 20, batch_size = 8, 
                    patience = 5, sent_similarity_threshold = 0.6, gnn_out_size = 768, num_heads = 8,
                    t5_learning_rates_dict = None, warmup_ratio = 0.1)
     """
     
     try:
          from t5_main import model_train_eval
          
     except ImportError as e:
          print(f"[ERROR] ImportError: {e}")
          traceback.print_exc()
          sys.exit(1)
          
     parser = argparse.ArgumentParser(description="Run T5 Model Training and Evaluation.")
     
     parser.add_argument('--dataset-path',
                         type=str,
                         required=True,
                         help='Path to the dataset containing all train/test/validation files.')

     parser.add_argument('--learning-rate',
                         type=float,
                         default=0.001,
                         help='Learning rate.')
     
     parser.add_argument('--num-epochs',
                         type=int,
                         default=20,
                         help='Number of training epochs.')

     parser.add_argument('--gnn-batch-size',
                         type=int,
                         default=32,
                         help='Batch size for GNN training and evaluation.')
     
     parser.add_argument('--llm-batch-size',
                         type=int,
                         default=4,
                         help='Batch size for LLM finetuning and evaluation.')
     
     parser.add_argument('--llm-accumulate-step',
                         type=int,
                         default=4,
                         help='For LLM gradient accumulation training.')
     
     parser.add_argument('--gnn-accumulation-steps',
                         type=int,
                         default=4,
                         help='For GNN gradient accumulation training.')
     
     parser.add_argument('--patience',
                         type=int,
                         default=5,
                         help='For EarlyStopper. How many epochs to stop training after lowest validation loss reach.')

     parser.add_argument('--sent-similarity-threshold',
                         type=float,
                         default=0.6,
                         help='Threshold for sentence similarity edge.')
     
     parser.add_argument('--gnn-out-size',
                         type=int,
                         default=768,
                         help='Embedding size of gnn model output.')
     
     parser.add_argument('--num-heads',
                         type=int,
                         default=8,
                         help='Number of head of gnn model.')
     
     parser.add_argument('--gnn-hidden-size',
                         type=int,
                         default=512,
                         help='Hidden layer size of gnn model.')
     
     ## T5 learning rate dict
     parser.add_argument('--t5-lr-shallow-layers',
                         type=float,
                         default=1e-4,
                         help='Learning rate for last 2 encoder/decoder blocks of T5.')
     parser.add_argument('--t5-lr-deep-layers',
                         type=float,
                         default=1e-5,
                         help='Learning rate for last 3/4 encoder/decoder blocks of T5.')
     parser.add_argument('--t5-lr-projector',
                         type=float,
                         default=1e-3,
                         help='Learning rate for the custom projector layer.')
     
     parser.add_argument('--warmup-ratio',
                         type=float,
                         default=0.1,
                         help='Wram-up ratio for T5 fine-tune scheduler.')
     

     args = parser.parse_args()
     
     t5_learning_rates_dict_from_args = {
          "shallow_layers": args.t5_lr_shallow_layers,
          "deep_layers": args.t5_lr_deep_layers,
          "projector": args.t5_lr_projector
     }
     
     print("--- Starting Model Training/Evaluation ---")
     print(f"Dataset Path:                {args.dataset_path}")
     print(f"Base Learning Rate:          {args.learning_rate}")
     print(f"Num Epochs:                  {args.num_epochs}")
     print(f"GNN Batch Size:              {args.gnn_batch_size}")
     print(f"LLM Batch Size:              {args.llm_batch_size}")
     print(f"Early Stopping Patience:     {args.patience}")
     print(f"LLM Accumulation Step:       {args.llm_accumulate_step}")
     print(f"Sentence Sim Threshold:      {args.sent_similarity_threshold}")
     print(f"GNN Output Size:             {args.gnn_out_size}")
     print(f"GNN Num Heads:               {args.num_heads}")
     print(f"GNN Hidden Size:             {args.gnn_hidden_size}")
     print(f"GNN Accumulation Step:       {args.gnn_accumulation_steps}")
     print(f"T5 Shallow LR (Last 2):      {args.t5_lr_shallow_layers}")
     print(f"T5 Deep LR (Deeper than 2):  {args.t5_lr_deep_layers}")
     print(f"T5 Projector LR:             {args.t5_lr_projector}")
     print(f"Scheduler Warmup Ratio:      {args.warmup_ratio}")
     print("-" * 33)
     
     try:
          scores = model_train_eval(
               dataset_path=args.dataset_path,
               learning_rate=args.learning_rate,
               num_epochs=args.num_epochs,
               gnn_batch_size=args.gnn_batch_size, 
               llm_batch_size=args.llm_batch_size,
               llm_accumulate_step=args.llm_accumulate_step,
               patience=args.patience,
               gnn_out_size = args.gnn_out_size,
               gnn_hidden_size = args.gnn_hidden_size,
               gnn_accumulation_steps=args.gnn_accumulation_steps,
               num_heads = args.num_heads,
               sent_similarity_threshold=args.sent_similarity_threshold, # Use underscore version here
               t5_learning_rates_dict=t5_learning_rates_dict_from_args,
               warmup_ratio=args.warmup_ratio
          )

          print("\n--- Evaluation Complete ---")
          print(f"Evaluated Scores:")
          for name, score in scores.items():
               print(f"{name}: {score}")

     except Exception as e:
          print(f"[ERROR] Error during model_train_eval: {e}")
          traceback.print_exc()
          sys.exit(1)

if __name__ == "__main__":
     run()