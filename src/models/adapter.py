from transformers import T5ForConditionalGeneration, AdapterConfig

# 加载 T5 模型
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# 定义适配器配置
adapter_config = AdapterConfig(
     mh_adapter=True,  # 在多头注意力模块中添加适配器
     output_adapter=True,  # 在输出层中添加适配器
     reduction_factor=16  # 下投影的降维比例
)

# 添加适配器
model.add_adapter("task_adapter", config=adapter_config)

# 激活适配器
model.train_adapter("task_adapter")

# 冻结原始模型的参数
model.freeze_model()