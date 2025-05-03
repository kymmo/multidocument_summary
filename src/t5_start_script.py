import argparse
import sys
import traceback

def run():
     """
     Main function to parse arguments and run the T5 training/evaluation.
     
     function signature: 
     def model_train_eval(dataset_path, learning_rate = 0.001,num_epochs = 20, batch_size = 8, 
                    patience = 5, sent_similarity_threshold = 0.6, 
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

     parser.add_argument('--batch-size',
                         type=int,
                         default=8,
                         help='Batch size for training and evaluation.')
     
     parser.add_argument('--patience',
                         type=int,
                         default=5,
                         help='For EarlyStopper. How many epochs to stop training after lowest validation loss reach.')

     parser.add_argument('--sent-similarity-threshold',
                         type=float,
                         default=0.6,
                         help='Threshold for sentence similarity edge.')
     
     ## T5 learning rate dict
     parser.add_argument('--t5-lr-encoder-last2',
                         type=float,
                         default=1e-4,
                         help='Learning rate for last 2 encoder blocks of T5.')
     parser.add_argument('--t5-lr-decoder-last2',
                         type=float,
                         default=1e-4,
                         help='Learning rate for last 2 decoder blocks of T5.')
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
          "encoder_last2": args.t5_lr_encoder_last2,
          "decoder_last2": args.t5_lr_decoder_last2,
          "projector": args.t5_lr_projector
     }
     
     print("--- Starting Model Training/Evaluation ---")
     print(f"Dataset Path:                {args.dataset_path}")
     print(f"Base Learning Rate:          {args.learning_rate}")
     print(f"Num Epochs:                  {args.num_epochs}")
     print(f"Batch Size:                  {args.batch_size}")
     print(f"Early Stopping Patience:     {args.patience}")
     print(f"Sentence Sim Threshold:      {args.sent_similarity_threshold}")
     print(f"T5 Encoder LR (Last 2):    {args.t5_lr_encoder_last2}")
     print(f"T5 Decoder LR (Last 2):    {args.t5_lr_decoder_last2}")
     print(f"T5 Projector LR:             {args.t5_lr_projector}")
     print(f"Scheduler Warmup Ratio:      {args.warmup_ratio}")
     print("-" * 20)
     
     try:
          scores = model_train_eval(
               dataset_path=args.dataset_path,
               learning_rate=args.learning_rate,
               num_epochs=args.num_epochs,
               batch_size=args.batch_size,
               patience=args.patience,
               sent_similarity_threshold=args.sent_similarity_threshold, # Use underscore version here
               t5_learning_rates_dict=t5_learning_rates_dict_from_args,
               warmup_ratio=args.warmup_ratio
          )

          print("\n--- Evaluation Complete ---")
          print(f"Evaluated ROUGE scores: \n {scores}")

     except Exception as e:
          print(f"[ERROR] Error during model_train_eval: {e}")
          traceback.print_exc()
          sys.exit(1)

if __name__ == "__main__":
     run()