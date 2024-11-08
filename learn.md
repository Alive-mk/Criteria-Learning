# 1. rejection sampling
## 1.1 背景介绍

拒绝采样是一种蒙特卡洛算法，用于借助代理分布从一个复杂的（“难以采样的”）分布中采样数据。

什么是蒙特卡洛？如果一个方法/算法使用**随机数**来解决问题，那么它被归类为蒙特卡洛方法。在拒绝采样的背景下，蒙特卡洛（也称为随机性）帮助实施算法中的标准。关于采样，几乎所有蒙特卡洛方法中存在的一个核心思想是，**如果你不能从你的目标分布函数中采样，那么使用另一个分布函数（因此被称为提议函数）。**

<img src="pictures\image-20241106182740289.png" alt="image-20241106182740289" style="zoom:50%;" />

然而，采样程序必须**“遵循目标分布”**。遵循“目标分布”意味着我们应该根据它们发生的可能性得到若干样本。简单来说，高概率区域的样本应该更多。

这也意味着，当我们使用一个提议函数时，我们必须引入必要的修正，以确保我们的采样程序遵循目标分布函数！

## 1.2 方法介绍

<img src="pictures\image-20241106185335076.png" alt="image-20241106185335076" style="zoom: 67%;" />

<img src="pictures\image-20241106185434856.png" alt="image-20241106185434856" style="zoom:67%;" />

## 1.3 具体例子

<img src="pictures\image-20241106185641675.png" alt="image-20241106185641675" style="zoom:67%;" />

<img src="pictures\image-20241106185750614.png" alt="image-20241106185750614" style="zoom:67%;" />





# 2. 示例代码理解

## 2.1 inference-score.py

- `for answer in conversation [-1] ['content']:`  //遍历访问最后一轮对话中的所有回答
- `sequence[start​ : end:step]` // 切片操作
- `chat.append({'role':"assistant", 'content':answer})` //将遍历的回复添加到chat中
- `with open(f'score/{DATASET}/{TYPE}/{MODEL}_1.json', 'w', encoding='utf-8') as json_output:    json.dump(score_list, json_output, ensure_ascii=False, indent=4)`  
          以写入模式"w"打开这个文件，表示如果这个文件已经存在将会覆盖。
  json.dump将score_list写入文件json_output中，ensure_ascii为false表示输出中允许非ascii符，这样可以更好的保留原始文本比如中文。
  indent=4用于格式化输出，缩进为四个空格，使json文件更具有可读性。

## 2.2 rejection_sampling.py

- `tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)`
  `AutoTokenizer`使transformer库中的一个类，用于自动加载与特定模式相匹配的分词器。分词器主要是用来将文本转换为模型所能理解的格式（比如将单词转换为对应的词汇索引）
  `from_pretrained`是用来从指定的路径或模型名称来加载的预训练的分词器
  `trust_remote_code`表示信任远程加载的代码

- 

  ```python
  model = AutoModelForCausalLM.from_pretrained(
  
       MODEL_PATH,
  
       torch_dtype=torch.bfloat16,
  
       low_cpu_mem_usage=True,
  
       device_map="balanced",
  
       trust_remote_code=True
  
    ).eval()
  ```

  `AutoModelForCausalLM`是一个transformer库的一个类，专用于因果语言模型，通常用于生成文本（GPT系列）
  `torch_dtype=torch.bfloat16` 半精度浮点格式常用语加速推理
  `low_cpu_mem_usage` 减少cpu的使用，避免内存溢出
  `device_map="balanced"` 表示模型的不同部分将均匀的分配到不同的GPU设备上
  `.eval()`  将模型设置为评估模式
  <img src="pictures\image-20241103134937756.png" alt="image-20241103134937756" style="zoom:67%;" />

-   

  ```python
   inputs = tokenizer(
  
         history_text,
  
         return_tensors="pt",
  
         truncation=True,
  
         padding=True,
  
         max_length=1024
  
       ).to("cuda")
  ```
  
  <img src="pictures\image-20241103162133668.png" alt="image-20241103162133668" style="zoom: 50%;" />
  
  



## 2.3 lmdeploy工具的使用

（https://lmdeploy.readthedocs.io/en/latest/llm/pipeline.html）//链接

**使用 `pipeline` 时通常不需要手动处理输入文本的分词和张量转换。这是因为 `pipeline` 封装了这些细节，简化了模型的使用。**

```python
from lmdeploy import pipeline

pipe = pipeline('internlm/internlm2_5-7b-chat')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

- 使用pipeline函数创建一个特定模型的管道
- 向管道传入一个包含两个字符串的列表，分别是用户的输入：
  - `'Hi, pls intro yourself'`：询问模型自我介绍。
  - `'Shanghai is'`：一个未完成的句子，期望模型补全。
- 但这个没有设置长度容易爆内存

```python
from lmdeploy import GenerationConfig, pipeline

pipe = pipeline('internlm/internlm2_5-7b-chat')

prompts = ['Hi, pls intro yourself', 'Shanghai is']

response = pipe(prompts,

         gen_config=GenerationConfig(

           max_new_tokens=1024,

           top_p=0.8,

           top_k=40,

           temperature=0.6

         ))
```

- **`gen_config=GenerationConfig(...)`**：这里创建了一个 `GenerationConfig` 对象来设置生成参数：
  - **`max_new_tokens=1024`**：指定模型生成的最大新令牌数为 1024。这表示生成的文本可以包含最多 1024 个新令牌。
  - **`top_p=0.8`**：使用核采样（nucleus sampling），表示从概率累积为 0.8 的候选词中进行采样。这有助于控制生成文本的多样性。
  - **`top_k=40`**：限制在生成时考虑的前 40 个最可能的候选词。这有助于提高生成文本的质量。
  - **`temperature=0.6`**：温度参数控制采样的随机性。较低的温度（如 0.6）会导致生成更加确定性和一致的文本，而较高的温度会增加多样性和创造性。

**要让模型根据上下文生成回复，你可以将上下文信息作为输入传递给模型，通常将整个对话历史包含在输入中**

```python
from lmdeploy import pipeline

//创建模型管道

pipe = pipeline('internlm/internlm2_5-7b-chat')

//定义对话历史

conversation_history = [
    {'role': 'user', 'content': 'Hi, pls intro yourself'},
    {'role': 'assistant', 'content': 'I am an AI language model designed to assist with various tasks.'},
    {'role': 'user', 'content': 'Shanghai is'},
]

//将对话历史格式化为输入

formatted_input = []
for message in conversation_history:
    formatted_input.append(f"{message['role']}: {message['content']}")

//生成模型的回复

response = pipe(['\n'.join(formatted_input)])

//打印响应

print(response)
```

<img src="pictures\image-20241103145556851.png" alt="image-20241103145556851" style="zoom:50%;" />

**据第一轮对话的历史文本回答第二轮中的用户问题**

```python
from lmdeploy import pipeline, GenerationConfig

#创建生成配置

gen_config = GenerationConfig(top_p=0.8, top_k=40, temperature=0.8, max_new_tokens=1024)

#初始化模型管道

pipe = pipeline('internlm/internlm2_5-7b-chat')

#第一轮对话

first_conversation = [
    {
        "role": "user",
        "content": "I am making mayonnaise, it was starting to thicken but now it has become runny and liquid again, is there any way to salvage it?"
    },
    {
        "role": "assistant",
        "content": "Yes, it's possible to fix runny mayonnaise! The most common reason for mayonnaise becoming runny is because the oil was added too quickly or the egg yolk wasn't emulsified properly. Here are some steps you can take to fix it: ..."
    }
]

#第二轮用户提问

second_user_question = {
    "role": "user",
    "content": "Why Aristotelian view of physics (impetus and stuff) is wrong?"
}

#构建输入，包含历史对话和新的用户提问

inputs = first_conversation + [second_user_question]

#生成回复

response = pipe([inputs], gen_config=gen_config)

#打印生成的回复

print(response)
```

<img src="pictures\image-20241103164916587.png" alt="image-20241103164916587" style="zoom: 67%;" />

## 2.4 vllm库

`vLLM` 是一个高性能的推理库，用于在大语言模型上进行高效的文本生成和推理。`LLM` 是该库中的核心类，用于加载和运行大语言模型，而 `SamplingParams` 是用于配置采样参数的类。这些工具结合起来能够更好地控制模型的生成过程。以下是详细的介绍和用法：

### 2.4.1 `LLM`类

<img src="pictures\image-20241106192710496.png" alt="image-20241106192710496" style="zoom:67%;" />

<img src="pictures\image-20241106192803205.png" alt="image-20241106192803205" style="zoom:67%;" />

### 2.4.2 `SamplingParams`类

<img src="pictures\image-20241106192937447.png" alt="image-20241106192937447" style="zoom:67%;" />

### 2.4.3 例子

```python
from vllm import LLM, SamplingParams

# Sample prompts.

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.

llm = LLM(model="facebook/opt-125m")

# Generate texts from the prompts. The output is a list of RequestOutput objects

# that contain the prompt, generated text, and other information.

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

Prompt: 'Hello, my name is', Generated text: ' John and I am a software developer.'
Prompt: 'The president of the United States is', Generated text: ' currently Joe Biden.'
Prompt: 'The capital of France is', Generated text: ' Paris, a city known for its art and culture.'
Prompt: 'The future of AI is', Generated text: ' full of exciting possibilities and potential advancements in various fields.'
```

以上为预计输出。

<img src="C:\Users\LX\Desktop\学习\Criteria-Learning\pictures\image-20241106193357798.png" alt="image-20241106193357798" style="zoom:67%;" />

# 3. 我的代码

## 3.1 基础知识

### 3.1.1 placehold_prompt

<img src="pictures\image-20241106194105186.png" alt="image-20241106194105186" style="zoom:67%;" />

```python
history = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "<PLACEHOLDER>"}
]

#假设调用生成函数后生成了文本

generated_text = "The capital of France is Paris."

#替换占位符

for message in history:
    if message["content"] == "<PLACEHOLDER>":
        message["content"] = generated_text

print(history)
```



### 3.1.2 `LLM()`

```python
llm = LLM(model=f"model/{MODEL}", trust_remote_code=True, tensor_parallel_size=GPU_NUM, enforce_eager=True)
```

<img src="C:\Users\LX\Desktop\学习\Criteria-Learning\pictures\image-20241106195655458.png" alt="image-20241106195655458" style="zoom:67%;" />

### 3.1.3 llm.chat()

```python
#调用 llm.chat() 生成对话内容
response = llm.chat(
    messages=messages,
    max_length=max_length,
    sampling_params=sampling_params,
    stop=stop,
    num_return_sequences=num_return_sequences
)
```

<img src="pictures\image-20241108110640553.png" alt="image-20241108110640553" style="zoom:67%;" />

### 3.1.4 output[0].outputs[0].text

```python
initial_output = llm.chat(messages=conversation, sampling_params=sampling_params, use_tqdm=False)
initial_answer = initial_output[0].outputs[0].text
```

<img src="pictures\image-20241108112011039.png" alt="image-20241108112011039" style="zoom:67%;" />

### 3.1.5 tokenizer.apply_chat_template(...)

```python
    text = llm_tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True
    )
```

​	用于将对话历史（`conversation`）通过特定模板转换为输入文本，以便传递给模型进行进一步的生成任务。

#### 3.1.5.1 `llm_tokenizer` 和 `Tokenizer` 的概念

`llm_tokenizer` 是一个 `Tokenizer` 对象，用于将自然语言文本转换为模型能够理解的格式（如 token ID）并处理输入文本的结构化工作。

- **`Tokenization`（标记化）**：将文本拆分为单个的基本单位，称为 "tokens"（标记），例如单词或子词。

- **Mapping to IDs（映射到 ID）**：将每个标记映射为特定的整数 ID，这样模型可以处理标记序列而不是原始文本。

- **Handling Special Tokens（处理特殊标记）**：引入特殊的开始、结束、填充等标记，帮助模型理解文本的结构。

#### 3.1.5.2 `conversation` 参数

`conversation` 是传递给 `apply_chat_template` 方法的参数，代表对话历史。它通常是一个包含多轮对话的列表，每一项都是一个字典，结构如下：

```python
conversation = [
    {"role": "user", "content": "What is quantum physics?"},
    {"role": "assistant", "content": "Quantum physics studies the behavior of particles at atomic scales..."}
]
```

#### 3.1.5.3 `apply_chat_template` 方法

`apply_chat_template` 是一个 `Tokenizer` 对象的方法，用于将对话历史应用到特定的模板中，从而生成一段适合输入模型的文本。这个方法的作用是将对话的格式化方式标准化，以确保模型在处理输入时能够识别对话的结构和顺序。**加入角色标记。**

#### `apply_chat_template` 的作用

- **格式化对话内容**：`apply_chat_template` 将 `conversation` 的内容格式化成模型预期的模板样式。例如，它可能会在每条消息前加入 "User:" 或 "Assistant:" 标签，使模型理解不同角色之间的对话。
- **加入生成提示**：通过参数 `add_generation_prompt=True`，可以在对话的末尾添加生成提示（例如 `Assistant:`），指示模型在此处生成新内容。

#### 3.1.5.4 `tokenize` 参数

```python
    text = llm_tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = llm_tokenizer([text], return_tensors="pt", padding=True).to("cuda")
```

**因为后续`tokenizer`要调用设置其他参数，所以前面`tokenize`设置为False没有立即token化。**

`tokenize` 参数用于控制是否在应用模板时立即将生成的文本进行 token 化（即转换为 tokens 或 token IDs）。在机器学习和自然语言处理任务中，`tokenize` 的主要功能如下：

- **`tokenize=False`**：表示不立即对文本进行 token 化，保持输出为原始文本格式。这种设置通常用于预处理阶段，需要进一步格式化文本，或准备传递给模型时再进行 token 化。
- **`tokenize=True`**：表示立即对文本进行 token 化，将文本直接转换为 token IDs 格式。适用于需要立即将文本输入模型的场景。

在某些对话系统中，将 `tokenize` 设置为 `False` 可以让用户在格式化文本后进行进一步处理（如检查格式），而在最终传递给模型时再进行 token 化。

## 3.2 `padding_side`报错

<img src="pictures\image-20241108153604163.png" alt="image-20241108153604163" style="zoom:67%;" />

<img src="pictures\image-20241108153644718.png" alt="image-20241108153644718" style="zoom:67%;" />

解决链接：

[padding_side报错解决](https://huggingface.co/THUDM/LongWriter-glm4-9b/commit/778b5712634889f5123d6c463ca383bc6dd5c621)



