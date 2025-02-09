import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

large_model = "facebook/bart-large"
small_model = "facebook/bart-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bart_tokenizer = BartTokenizer.from_pretrained(small_model)
bart_model = BartForConditionalGeneration.from_pretrained(small_model).to(device)

def get_bart_outputs(gnn_sent_embeddings, sample_node_sent_maps, summary_length = 200, sequence_length = 512):
     sent_list = []
     for doc_node_sent_map in sample_node_sent_maps:
          for id in range(len(doc_node_sent_map)):
               sent_list.append(doc_node_sent_map[id])
     
     ## get BART embeddings
     inputs = bart_tokenizer(sent_list, return_tensors="pt", padding=True, truncation=True)
     input_ids = inputs['input_ids']
     attention_mask = inputs['attention_mask']

     bart_model.eval()
     with torch.no_grad():
          encoder_outputs = bart_model.encoder(input_ids, attention_mask=attention_mask)

     bart_embeddings = encoder_outputs.last_hidden_state
     # ignore paddding
     masked_embeddings = bart_embeddings * attention_mask.unsqueeze(-1)
     avg_bart_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True) ## (sentence_number, embedding)
     
     ## project GNN embeddin to BART space
     out_size = gnn_sent_embeddings.shape[1]
     BART_embed_projector = nn.Linear(out_size, bart_model.config.d_model)
     BART_embed_projector.load_state_dict(torch.load('bart_projector_weights.pth', weights_only=True))
     projected_gnn_embeddings = BART_embed_projector(gnn_sent_embeddings)
     
     ## combine GNN and BART embedding
     combined_embeddings = projected_gnn_embeddings + avg_bart_embeddings
     
     comb_embed_size = combined_embeddings.shape
     ##reshape
     sample_length = comb_embed_size[0]
     embedding_length = comb_embed_size[1]
     batch_size = sample_length // sequence_length
     remaining = sample_length % sequence_length

     if remaining != 0:
          padding_size = sequence_length - remaining
          padding_tensor = torch.zeros(padding_size, embedding_length) ## padding by zeroes
          combined_embeddings = torch.cat([combined_embeddings, padding_tensor], dim=0)

     reshaped_tensor = combined_embeddings.view(batch_size + 1, sequence_length, embedding_length)
     summary_attention_mask = torch.ones(batch_size + 1, sequence_length)
     
     encoder_outputs = bart_model.encoder(
          attention_mask=summary_attention_mask,
          inputs_embeds=reshaped_tensor
     )
     
     decoder_input_ids = bart_tokenizer("summarize:", return_tensors="pt").input_ids

     output = bart_model.generate(
          decoder_input_ids=decoder_input_ids,
          encoder_outputs=encoder_outputs,
          attention_mask=summary_attention_mask,
          max_length=summary_length,
          num_beams=3,
          no_repeat_ngram_size=2
     )

     decoded_output = bart_tokenizer.decode(output[0], skip_special_tokens=True)
     
     return decoded_output
