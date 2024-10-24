import torch
from transformers import AutoModel, AutoTokenizer
import os
import json

# 限制只使用 GPU 0-4
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# 清空缓存
torch.cuda.empty_cache()

# 检查 GPU 的可用性及其名称
print("可用的 GPU 数量:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 加载模型并分配到多个 GPU
model = AutoModel.from_pretrained(
    "../model/internlm2-20b-reward",
    max_memory={i: '20GB' for i in range(torch.cuda.device_count())},
    torch_dtype=torch.float16,
    device_map="balanced", 
    trust_remote_code=True
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("../model/internlm2-20b-reward", trust_remote_code=True)

# 读取 JSON 文件
with open("output_data/sampled_data_temp_0.5.json", "r") as f:
    data = json.load(f)

# 将 JSON 数据转换为模型输入格式，并在分词时设置最大长度和截断
def convert_to_model_input(conversation, max_length=128):
    # 构建适合分词器的格式
    truncated_conversation = []
    
    for message in conversation:
        role = "user" if message["from"] == "human" else "assistant"
        content = message["value"]
        
        # 对内容进行分词并截断
        tokenized = tokenizer(
            content,
            max_length=max_length,  # 设置最大长度
            truncation=True,        # 启用截断
        )
        
        # 将截断后的 token IDs 转回原始文本
        truncated_text = tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True)
        
        # 保留 role 并更新截断后的内容
        truncated_conversation.append({"role": role, "content": truncated_text})
    
    return truncated_conversation

# 转换所有对话数据，并在转换时进行截断处理
chats = [convert_to_model_input(item["conversations"]) for item in data]

score = model.get_score(tokenizer, chats[0])

print(score)

# 打印每个 GPU 的显存使用情况
for i in range(torch.cuda.device_count()):
    print(f"GPU {i} 已用显存: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
    print(f"GPU {i} 剩余显存: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
