from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LongT5ForConditionalGeneration

lt5_tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
lt5_model = (
     LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base")
     .to("cuda")
     .half()
)

def t5_fine_tune(gnn_sent_embeddings, node_sent_maps):
     
     
     inputs_dict = lt5_tokenizer(
          batch["article"], max_length=16384, padding="max_length", truncation=True, return_tensors="pt"
     )
     input_ids = inputs_dict.input_ids.to("cuda")
     attention_mask = inputs_dict.attention_mask.to("cuda")
     output_ids = lt5_model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=2)
     output = lt5_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
     
     # TODO: adapter
     


# # 定义训练参数
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",  # 每个 epoch 评估一次
#     learning_rate=5e-5,  # 小学习率
#     per_device_train_batch_size=8,  # 小批量大小
#     per_device_eval_batch_size=8,
#     num_train_epochs=10,  # 少量数据时可以增加 epoch
#     weight_decay=0.01,  # 权重衰减（正则化）
#     save_total_limit=2,  # 只保存最后两个模型
#     save_steps=500,
#     logging_dir="./logs",
#     logging_steps=10,
#     fp16=True,  # 使用混合精度训练
#     load_best_model_at_end=True,  # 早停
# )