1. Message、CompletionPrediction、ChatPrediction是三个类型化字典，用于定义对话中的单条消息（输入输出）数据结构

   

2. llama.build()函数用于模型的初始化

   

3. generate（） 生成文本，控制生成

   

3. text_completion() 单文本任务

   

4. chat_completion()多文本任务，对话式的

   ![image-20250309143359729](assets/image-20250309143359729.png)

5. sample_top_p 核采样    从概率分布中选择下一个生成的 token。Top-p 采样的核心思想是保留概率最高的 token，直到累积概率超过阈值 p，然后从这些 token 中随机采样。主要是其到一个平衡大模型输出文本的目的，当生成结果过于创新 降低top_p,当生成结果缺乏创意 提高top_p.

![image-20250309112850812](assets/image-20250309112850812.png)





## **模型初始化的build（）函数**

**1.检查分布式训练是否初始化，**没有，则使用 NCCL (NVIDIA Collective Communications Library) 后端初始化分布式进程组

**2.检查是否进行了模型并行初始化**  没有，则使用模型并行初始化，如果没有指定 model_parallel_size，则从环境变量 WORLD_SIZE（总进程数） 中获取（如果不存在，则默认为1)

**3.获取本地进程的GPU编号，设置为当前进程的GPU设备，默认为0**

**4.设置随机种子**，确保所有进程的随机初始化一致

**5.只允许主进程 local_rank=0 输出信息到控制台**，其余进程则输出信息到/dev/null(丢弃输出)

避免多进程重复输出导致混乱，减少不必要的输出开销

#### 加载模型的检查点文件和参数配置

6.记录模型开始时间

7.获取ckpt_dir(检查点)目录下的所有path文件，按照字母顺序进行排序

8.检查是否存在检查点文件，如果检查点文件不存在，则抛出错误

9.检查模型并行进程数是否与检查点文件数量匹配，如果不匹配，则抛出错误

10.取当前进程对应的检查点文件路径

11.将检查点文件加载到CPU上

12.加载检查点文件路径下的params.json文件，并将文件从json字符串*转换为python字典*

#### 完成模型的最终初始化和加载过程，并打印加载模型花费的时间

**13.将模型参数转换为ModelArgs类型（定义模型参数的类），将所有参数整合到一个MOdelArgs对象中**

14.初始化分词器，将tokenizer_path路径下的文件加载到Tokenizer对象中

15.将Tokenizer对象中的n_words属性赋值给model_args.vocab_size，即模型参数中的词汇量大小

16.设置默认的tensor类型为torch.cuda.HalfTensor，即使用半精度浮点数进行计算

17.创建Transformer对象，将model_args作为参数传递给Transformer的构造函数

19.将检查点文件中的模型状态加载到Transformer对象中，strict=False表示忽略不匹配的参数

20.创建并返回Llama类的实例，将model和tokenizer作为参数传递给Llama的构造函数

21.init 初始化Llama类的实例，将model和tokenizer作为参数传递给Llama的构造函数





## 处理多轮对话的流程

### chat_completion()函数

**1.调用chat_completion()函数：**传入相应参数，主要是dialogs参数，它是指对话的列表，列表格式要遵守message定义的格式（原因：dialogs的数据格式List[List[Message]]嵌套列表结构，表示多个对话）

假设

```python
dialogs=[

{"role":"systgem"|“user"|"assistant"  #消息角色

"content":str   #消息内容

},

...

]
```

**2.检查是否含有特殊标记：**遍历消息列表的每一条消息，检查是否含有特殊标签（指令和系统消息的开始和结束标记），如果含有则将结果（True）添加到unsafe_requests的列表中。

**3.检查第一条消息是否是系统消息：**如果是，则将系统消息与下一条（用户）消息进行合并，系统消息包含开始、结束提示符。（合并原因：llama2的对话模板将系统消息和用户消息合并为了一个整体，将其作为整体进行输入，使模型更容易理解上下文，防止模型出现混淆）。

**4.断言检查偶数为user，奇数为assistant**：多轮对话依次为：user-assistant-user-...顺序不可改变，若顺序不对，则抛出错误。

**5.将格式化的字符串编码为Token序列**：首先提取用户和大模型的消息，对字符串进行格式化，然后再使用Token分词器编码问Token序列，并sum合并到一起。（List[int]的原因：Token序列其实就是大模型可以理解的数字序列，即每个字都有唯一的整数（token ID)）。

**6.断言确保最后一条消息是用户消息，同时将最后一条用户消息添加到dialog_tokens的Token序列中**（最后一条消息必须是用户消息的原因：如果最后一条消息是助手消息，那么接下来模型将不知道该回复什么，缺少用户消息）（最后一条用户消息需要单独添加的原因：zip是将可迭代对象的元素按位置进行配对，而最后一条用户消息还没有生成相应的AI回复，zip无法获取到他，所以要但对添加）

**7.将当前的对话Tokens序列添加到prompt_tokens中**，原因：可以批量处理多个对话任务，使模型一次性生成所有回复。

**8.调用generate()函数**，将处理后的prompt_Token序列(即用户和助手的完整对话历史)传递给generate()函数，返回generation_tokens(助手对用户消息的最新回复的Tokens序列)和generation_logprobs(每个token的对数概率)



### generate()函数的执行过程

1. **pad_id 获取模型的填充tokenID，一般为唯一的一个整数**(作用：在批次处理过程中，所有序列必须有相同的长度，pad_id为长度较短的序列进行填充，在后续模型计算中，填充的token通过掩码机制会被忽略，避免产生影响)

   > - 序列1：`[1, 2, 3]`（长度为3）
   >
   > - 序列2：`[4, 5, 6, 7]`（长度为4）
   > - 填充后：
   >   - 序列1：`[1, 2, 3, pad_id]`
   >   - 序列2：`[4, 5, 6, 7]`

2. **创建一个初始化为pad_id的tokens。将所有的提示tokens序列转化为Tensor形式，依次添加到tokens张量中进行存储。**此时tokens是经过分词和编码的tokens序列。**如果需要计算词元的对数概率，则再创建一个二维张量.**

3. **创建**与bsz形状相同的张量eos_reached，用来存储每段对话是否到达EOS结束。**创建**与tokens形状相同的张量input_text_mask，判断tokens中的元素是否和pad_id相同。

4. **当最小提示token长度等于最大token长度时，没有空间生成新的Token了，此时便无需循环，直接计算logits（*模型的下一个 Token 的预测结果*）**（forward前向传播的流程：就收参数tokens序列和当前位置信息，输出`logits` 表示模型对每个位置（`seq_len`）的每个 Token（`vocab_size`）的预测分数（未经过 Softmax 归一化的原始分数）)。**通过F.cross_entropy函数返回每个Token的交叉熵损失（即对数概率的负数），取负号得到每个Token的对数概率**，token_logprobs 是一个形状为 (batch_size, seq_len) 的张量，表示每个 Token 的对数概率。反映了模型对每个 Token 的预测置信度,评估模型的预测质量。

   ![image-20250311121659210](assets/image-20250311121659210.png)

5. **当最小提示token长度小于最大token长度的时候，循环从最短提示长度开始，到最大token长度结束，调用forward函数生成logits**

6. **如果当温度大于0，则使用softmax函数计算概率，并进行top-p采样** 因为模型是对下一个Token的预测分数，因此提取最后一个位置的logits（logits[: -1])    dim=-1 表示在最后一个维度（即 vocab_size 维度）上计算 Softmax。将logits转化为概率分布，然后调用sample_top_p函数来进行top_p采样。在累积概率超过p的token序列中随机选择一个，返回他的原始索引。

   

7. **如果温度等于0，则此时的logits的元素值就代表了概率，直接取在最后一个位置使login达到最大值的索引即可。**

8. 展平维度，从（batch_size,1)（二维，表示每个样本的下一个Token的索引）展平为一维

   > `argmax` 的 `next_token` 是 **一维张量** `(batch_size,)`。
   >
   > `sample_top_p` 的 `next_token` 是 **二维张量** `(batch_size, 1)`。

9. **根据 `input_text_mask[:, cur_pos]` 的值选择是保留原 Token 还是使用新生成的 Token。**`input_text_mask` 用于标记哪些位置是真实文本，哪些位置是填充 Token。如果当前位置是真实文本，则保留原 Token；否则，使用新生成的 Token。（原因：确保在生成过程中，填充 Token 的位置被新Token覆盖，同时真实文本的位置保持不变。）

10. **将修改后的tokens放回原始位置。**

11. **如果需要则计算对数概率**，还是使用cross_entropy来计算，因为计算的是每个输入token预测下一个token的概率，所以要加一，位置一的预测目标是位置二的token

12. 判断生成的token是否是结束符，如果是则标记为true，更新prev_pos，进行下一次迭代。如果所有对话都达到了EOS,则停止生成。

13. 如果要计算对数概率，则将都对数概率从tensor张量转化为Python列表。初始化输出的token和对数概率列表。遍历tokens张量，将每个对话任务中的token和logprobs 添加到out_tokens和out_logprobs列表中

14. echo为true，则从0开始截取，及包含提示词元

15. 截取最终输出序列

16. 如果要计算对数概率，则从token_logprobs中截取最终的输出序列。

17. 如果包含结束符，则截取到EOS的位置。

18. 将截取后的token和probs添加到out_tokens和out_logprobs列表中。返回最终的输出token和logprobs




**9.根据生成的tokens和logprobs来输出，**对输出的整个token序列解码为文本，判断是否包含特殊标签，将tokens序列中的每个token进行解码，（方便后续使用?),输出每个token的对数概率





