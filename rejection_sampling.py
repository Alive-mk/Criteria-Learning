import json
import numpy as np
import random

# 读取 JSON 数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 定义对 "Temtem" 相关的拒绝采样逻辑
def is_talk_about_temtem(conversation):
    # 如果对话中有提到 "Temtem"，则返回 True
    for convo in conversation['conversations']:
        if 'temtem' in convo['value'].lower():
            return True
    return False

# 拒绝采样函数，考虑 temperature 参数
def rejection_sampling(data, temperature=1.0):
    selected_samples = []
    for conversation in data:
        if is_talk_about_temtem(conversation):
            # 使用 temperature 来控制是否接受样本
            accept_probability = np.exp(1 / temperature)
            if random.random() < accept_probability:
                selected_samples.append(conversation)
    return selected_samples

# 控制 temperature 参数进行多次采样
def run_sampling_with_temperature(data, temperature_values):
    results = {}
    for temp in temperature_values:
        print(f"Sampling with temperature = {temp}")
        sampled_data = rejection_sampling(data, temperature=temp)
        results[temp] = sampled_data
        print(f"Number of samples collected with temperature {temp}: {len(sampled_data)}")
    return results

# 主程序
if __name__ == "__main__":
    # 加载数据
    data_path = 'data/original_data.json'
    data = load_data(data_path)

    # 设置不同的 temperature 参数
    temperature_values = [0.5, 1.0, 1.5, 2.0]

    # 运行采样并观察 temperature 的影响
    sampled_results = run_sampling_with_temperature(data, temperature_values)

    # 输出结果到文件
    for temp, samples in sampled_results.items():
        output_file = f"output_data/sampled_data_temp_{temp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=4)
        print(f"Sampled data with temperature {temp} saved to {output_file}")
