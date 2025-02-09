import torch
import torch.nn as nn
from transformers import EncoderDecoderCache

from models.gnn_train_t5 import t5_tokenizer, t5_model

def get_t5_outputs(gnn_sent_embeddings, sample_node_sent_maps, summary_length = 200, sequence_length = 512):
     sent_list = []
     for doc_node_sent_map in sample_node_sent_maps:
          for id in range(len(doc_node_sent_map)):
               sent_list.append(doc_node_sent_map[id])
     
     ## get T5 embeddings
     inputs = t5_tokenizer(sent_list, return_tensors="pt", padding=True, truncation=True)
     input_ids = inputs['input_ids']
     attention_mask = inputs['attention_mask']

     t5_model.eval()
     with torch.no_grad():
          encoder_outputs = t5_model.encoder(input_ids, attention_mask=attention_mask)

     t5_embeddings = encoder_outputs.last_hidden_state
     # ignore paddding
     masked_embeddings = t5_embeddings * attention_mask.unsqueeze(-1)
     avg_t5_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True) ## (sentence_number, embedding)
     
     ## project GNN embeddin to T5 space
     out_size = gnn_sent_embeddings.shape[1]
     T5_embed_projector = nn.Linear(out_size, t5_model.config.d_model)
     T5_embed_projector.load_state_dict(torch.load('t5_projector_weights.pth', weights_only=True))
     projected_gnn_embeddings = T5_embed_projector(gnn_sent_embeddings)
     
     ## combine GNN and T5 embedding
     combined_embeddings = projected_gnn_embeddings + avg_t5_embeddings
     
     comb_embed_size = combined_embeddings.shape
     ##reshape to input T5
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
     
     encoder_outputs = t5_model.encoder(
          attention_mask=summary_attention_mask,
          inputs_embeds=reshaped_tensor
     )
     
     decoder_input_ids = t5_tokenizer("summarize:", return_tensors="pt").input_ids
     encoder_cache = EncoderDecoderCache(
          encoder_hidden_states=encoder_outputs.last_hidden_state,
          encoder_attention_mask=summary_attention_mask
     )
     
     output = t5_model.generate(
          decoder_input_ids=decoder_input_ids,
          # encoder_outputs=encoder_outputs,
          encoder_cache=encoder_cache,
          attention_mask=summary_attention_mask,
          max_length=summary_length,
          num_beams=3,
          no_repeat_ngram_size=2,
          # past_key_values=EncoderDecoderCache(),  # for transformers new version
          use_cache=True
     )

     decoded_output = t5_tokenizer.decode(output[0], skip_special_tokens=True)
     
     return decoded_output




###TODO: fine-tune