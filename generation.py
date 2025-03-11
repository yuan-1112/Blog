#是文本生成的核心文件，主要负责处理模型的推理和生成过程
'''
主要用途：
1.加载预训练的llama模型
2.处理输入文本和对话
3.控制生成过程（温度、采样等）
4.提供文本补全和对话生成接口
5.提供文本补全和对话生成接口
'''


import json   #处理json格式数据，用于读取模型参数文件params.json
import os       #os 提供操作系统相关功能，如环境变量读取，文件路径操作
import sys      #sys 提供python运行时系统的相关功能，如输入输出流
import time     #time 提供时间相关功能，用于计算模型加载时间
from pathlib import Path   #path 提供面向对象的文件系统路径处理
from typing import List, Literal, Optional, Tuple, TypedDict   #类型提示相关

#Pytorch相关导入
import torch    #Pytorch 深度学习框架的主要模块
import torch.nn.functional as F   #Pytorch的函数式接口，提供各种神经网络操作，神经网络相关的函数

# fairscale是一个用于大规模分布式训练的库
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,    #获取模型并行时的当前进程序号
    initialize_model_parallel,  #初始化模型并行环境
    model_parallel_is_initialized,  #检查模型并行时候已经初始化
)

#本地模块导入
from llama.model import ModelArgs, Transformer  #导入模型相关类 modelArgs,Transformer是模型的定义
from llama.tokenizer import Tokenizer  #导入分词器相关类，Tokenizer是用来处理文本的编码和解码

#literal类型是用来限定这个变量只能接收特定的值，即"system", "user", "assistant"
'''
system:系统消息，用于设置对话的上下文或者规则
user:用户消息，用于输入用户的问题或者请求
assistant:AI助手的回复消息
'''
Role = Literal["system", "user", "assistant"]

#Message、CompletionPrediction、ChatPrediction是三个类型化字典，用于定义对话中的单条消息（输入输出）数据结构

#规范化消息的格式 定义了消息的数据类型
#TypedDict 是一个类型化字典，该类用来定义对话中的单条消息结构
#role:消息的类型，可以是system,user,assistant
#content:消息的内容，是一个字符串
'''例子：
message = {
    "role": "user",
    "content": "你好，请问今天天气如何？"
}
'''
class Message(TypedDict):
    role: Role
    content: str

#定义了文本生成的预测结果的数据类型
'''
TypeDict 类似literal，为字典的每个键指定一个具体的类型。，不可替换，进行类型检查
total则表示类中的变量可以为空，即tokens和logprobs是可选字段
'''
class CompletionPrediction(TypedDict, total=False):
    #生成的文本和token都是字符串，而logrpobs是浮点数
    generation: str     #生成的文本内容
    tokens: List[str]  # 可选字段，生成文本的Token,词元列表
    logprobs: List[float]  # 可选字段 每个token的生成概率的对数值



#定义了聊天对话生成的预测结果的数据类型
class ChatPrediction(TypedDict, total=False):
    generation: Message 
    tokens: List[str]  #可选字段
    logprobs: List[float]  #可选字段

#表示对话的类型是一个消息列表
Dialog = List[Message]
'''
指令是指用户的输入，模型给用户的回复，帮助模型识别出文本中的指令部分
系统消息是全局的行为规则，用于设置整个上下文和基调
B_INST和E_INST这两个常量被用作指令的开始和结束标记。在这个上下文中，指令可能是指用户给模型的指令，
或者模型给用户的回复，帮助模型识别出文本中的指令部分
B_SYS和E_SYS这两个常量被用作系统消息的开始和结束标记。系统消息通常用于设置对话的上下文，或者给模型提供一些特定的指示
这些标记通常用于在对话中插入系统消息，以便模型能够正确处理和生成回复
'''
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


#SPECIAL_TAGS 收集所有用于格式化的特殊标记，这些标记在对话中用于表示特定的含义或功能，检查用户输入是否包含这些标记
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
#当用户输入中包含这些特殊表记时显示的错误消息，防止用户通过注入特殊标记来操纵模型的行为
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

#该类是代码的核心，包含了模型的初始化，文本生成和聊天对话生成等功能。
class Llama:
    @staticmethod   #静态方法，不需要实例化类就可以调用（如 Llama.build(...)）。
    #build方法用于构建Llama类的实例，接收多个参数，用于初始化模型和分词器
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,  #为什么可以默认为空，是因为下面的并行初始化的判断吗？
        seed: int = 1,   
    ) -> "Llama":
        """
        通过初始化和加载预训练模型来构建 Llama 实例。

        接收的参数：
            ckpt_dir (str)：包含检查点文件的目录路径
            tokenizer_path (str)：分词器文件的路径
            max_seq_len (int)：输入文本的最大序列长度
            max_batch_size (int)：推理时的最大批处理大小
            model_parallel_size (Optional[int], optional)：模型并行进程数
                如果未提供，将从环境变量中确定。默认为 None。
            seed 设置随机数生成器的种子值 确保每次运行程序时这些随机行为的结果相同

        返回：
            Llama：加载了模型和分词器的 Llama 类实例

        异常：
            AssertionError：如果指定目录中没有检查点文件，
                或者模型并行大小与检查点文件数量不匹配时抛出。

        注意：
            此方法初始化分布式进程组，设置 CUDA 设备，
            并加载预训练模型和分词器。
        """
        #检查是否进行了分布式训练的初始化
        #没有，则使用 NCCL (NVIDIA Collective Communications Library) 后端初始化分布式进程组
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        #检查是否进行了模型并行初始化
        #没有，则使用模型并行初始化
        if not model_parallel_is_initialized():
            #如果没有指定 model_parallel_size，则从环境变量 WORLD_SIZE（总进程数） 中获取（默认为1）
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

#在分布式训练中合理分配GPU资源，确保每个进程都使用独立的GPU
        #获取当前进程在本地的 GPU 编号
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        #将当前进程绑定到目前进行分布式训练的 GPU 设备
        torch.cuda.set_device(local_rank)


        # 设置了随机数生成器的种子，以确保所有进程生成的随机数是一致的
        torch.manual_seed(seed)

        #只允许主进程 local_rank=0 输出信息到控制台，其余进程则输出信息到/dev/null(丢弃输出)
        #避免多进程重复输出导致混乱，减少不必要的输出开销
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")


#加载模型的检查点文件（checkpoint）和参数配置。
        #记录模型加载开始时间
        start_time = time.time()
        #获取ckpt_dir(检查点)目录下的所有path文件，并按照字母顺序排序
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        #检查是否存在检查点文件，如果检查点文件不存在，则抛出错误
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        #检查模型并行进程数是否与检查点文件数量匹配，如果不匹配，则抛出错误
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        #获取当前进程对应的检查点文件路径
        #get_model_parallel_rank() 获取当前进程的模型并行排名，即在分布式训练中当前进程的编号
        #checkpoints[get_model_parallel_rank()] 获取当前进程对应的检查点文件路径
        ckpt_path = checkpoints[get_model_parallel_rank()]
        #加载检查点文件，将文件加载到CPU上
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        #打开params.json文件，读取模型参数，json.loads()将json字符串转换为python字典
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())


#完成了模型的最终初始化和加载过程，并打印加载模型花费的时间
        #将模型参数转换为ModelArgs类型，ModelArgs是定义模型参数的类，将所有参数整合到一个MOdelArgs对象中
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,       #输入文本的最大序列长度
            max_batch_size=max_batch_size, #推理时的最大批处理大小
            **params,                      #将params字典中的参数传递给ModelArgs
        )

        #初始化分词器，将tokenizer_path路径下的文件加载到Tokenizer对象中
        tokenizer = Tokenizer(model_path=tokenizer_path)
        #将Tokenizer对象中的n_words属性赋值给model_args.vocab_size，即模型参数中的词汇量大小
        model_args.vocab_size = tokenizer.n_words
        #设置默认的tensor类型为torch.cuda.HalfTensor，即使用半精度浮点数进行计算
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        #创建Transformer对象，将model_args作为参数传递给Transformer的构造函数
        model = Transformer(model_args)
        #将检查点文件中的模型状态加载到Transformer对象中，strict=False表示忽略不匹配的参数
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        #创建并返回Llama类的实例，将model和tokenizer作为参数传递给Llama的构造函数
        return Llama(model, tokenizer)
    
    #初始化Llama类的实例，将model和tokenizer作为参数传递给Llama的构造函数
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    #generate方法用于生成文本序列，接收多个参数，用于控制生成过程
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        使用语言生成模型基于提供的提示生成文本序列。

        参数：
            prompt_tokens (List[List[int]])：分词后的提示列表，每个提示表示为整数列表
            max_gen_len (int)：生成文本序列的最大长度
            temperature (float, optional)：用于控制采样随机性的温度值。默认为 0.6
            top_p (float, optional)：核采样的概率阈值。默认为 0.9
            logprobs (bool, optional)：是否计算词元的对数概率。默认为 False
            echo (bool, optional)：是否在生成输出中包含提示词元。默认为 False

        返回：
            Tuple[List[List[int]], Optional[List[List[float]]]]：
            包含生成的词元序列，如果 logprobs 为 True，则包含对应的词元对数概率

        注意：
            此方法使用提供的提示作为生成文本的基础。它使用核采样来产生具有可控随机性的文本。
            如果 logprobs 为 True，将计算每个生成词元的对数概率。
        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)


#计算了提示的最小和最大长度，并确保最大长度不超过最大序列长度。
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len

        #最大序列长度，最大生成长度加上最大提示长度之间的较小值
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        #获取分词器的填充token ID
        pad_id = self.tokenizer.pad_id

        #创建一个全部填充为pad_id的2D张量
        '''
        tokens = torch.full(
        size=(bsz, total_len),  # 形状：(批次大小, 最大序列长度)
        fill_value=pad_id,      # 用pad_id填充
        dtype=torch.long,       # 数据类型：长整型
        device="cuda"           # 放在GPU上
        )
        '''
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        #遍历prompt_tokens，将每个提示的token填充到tokens张量中
        for k, t in enumerate(prompt_tokens):
            #将提示的token填充到tokens张量中，从tokens的第k行开始，填充到提示的长度
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        #如果需要计算词元的对数概率，则创建一个与tokens形状相同的张量，用于存储每个词元的对数概率
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        
        '''
        # 假设有两个序列（bsz=2）：
        tokens = torch.tensor([
            [1, 2, 3, 0, 0],  # 0是pad_id
            [4, 5, 0, 0, 0]
        ])

        # 则input_text_mask会是：
        input_text_mask = [
            [True,  True,  True,  False, False],  # 序列1
            [True,  True,  False, False, False]   # 序列2
        ]

        # eos_reached初始化为：
        eos_reached = [False, False]  # 表示两个序列都还没结束
        '''
        #初始化prev_pos为0，用于记录当前处理的位置
        prev_pos = 0
        #创建一个与bsz大小相同的布尔型tensor，用于存储每个样本是否已经到达EOS（结束符）
        # 例如，如果bsz=3，则创建：[False, False, False]
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        #创建一个与tokens形状相同的布尔张量，用于标记哪些位置是填充的
        ## True表示实际token，False表示填充token
        input_text_mask = tokens != pad_id

        #如果最小提示长度等于最大序列长度，则直接计算logits
        #输入的提示（prompt_tokens）长度已经达到最大序列长度（total_len）时，模型 ​没有空间生成新的 Token，
        # 因此直接计算 logits 并返回结果，而无需进入生成循环。（最短的提示已经占满了最大序列长度，此时 ​所有提示（最长的提示）都没有空间生成新的 Token
        if min_prompt_len == total_len:
           #​调用模型的前向传播函数，基于当前的输入 Token 序列（tokens）和位置信息（prev_pos），计算模型的下一个 Token 的预测结果（logits）
            logits = self.model.forward(tokens, prev_pos)
            #计算每个token的对数概率，使用交叉熵损失函数，忽略填充token
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),   #将logits的维度从(batch_size, seq_len, vocab_size)转换为(batch_size, vocab_size, seq_len),转置使形状符合cross_entropy要求
                target=tokens,                  #目标token序列
                reduction="none",              #不进行平均或求和，返回每个样本的损失
                ignore_index=pad_id,            #忽略填充token的损失
            )
        #如果最小提示长度小于总长度
        # 从最短提示长度开始，到总长度结束
        # 每次生成一个新的token
        for cur_pos in range(min_prompt_len, total_len):
            #前向传播计算logits，tokens[:, prev_pos:cur_pos]是当前处理的token序列，prev_pos是当前处理的位置
            '''
            # tokens 是一个二维张量，形状为 [batch_size, sequence_length]
            # 例如：
            tokens = [
                [1, 2, 3, 4, 0],  # 第一个序列
                [5, 6, 7, 8, 0]   # 第二个序列
            ]

            # [:, prev_pos:cur_pos] 的含义：
            # : 表示保留第一维的所有内容（所有序列）
            # prev_pos:cur_pos 表示在第二维上取从prev_pos到cur_pos-1的部分

            # 如果 prev_pos=1, cur_pos=3，则结果为：
            result = [
                [2, 3],    # 第一个序列的第2、3个token
                [6, 7]     # 第二个序列的第2、3个token
            ]
            '''
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)   # 选取从prev_pos到cur_pos的token序列
            #如果温度大于0，则使用softmax函数计算概率，并进行top-p采样
             # temperature越高，分布越平缓（更随机）
             # temperature越低，分布越尖锐（更确定）
            if temperature > 0:
                '''
                # logits 的形状通常是 [batch_size, sequence_length, vocab_size]
                # [:, -1] 的含义：
                # : 保留第一维（所有批次）
                # -1 取最后一个位置的预测

                # 例如：
                logits = [
                    # 第一个序列
                    [
                        [0.1, 0.2, 0.3],  # 第1个位置的预测
                        [0.4, 0.5, 0.6],  # 第2个位置的预测
                        [0.7, 0.8, 0.9]   # 第3个位置的预测（最后一个位置）
                    ],
                    # 第二个序列
                    [
                        [1.1, 1.2, 1.3],
                        [1.4, 1.5, 1.6],
                        [1.7, 1.8, 1.9]   # 最后一个位置
                    ]
                ]

                # logits[:, -1] 会得到：
                result = [
                    [0.7, 0.8, 0.9],  # 第一个序列的最后一个位置
                    [1.7, 1.8, 1.9]   # 第二个序列的最后一个位置
                ]
                '''
                # 因为我们只需要预测下一个token
                # 模型输出了所有位置的预测，但我们只关心最后一个位置
                # 这个位置的预测将用于生成下一个token
                '''
                # 假设有一个3维张量，形状为 [batch_size, sequence_length, vocab_size]
                # dim=0：第一维（批次维度）
                # dim=1：第二维（序列长度维度）
                # dim=2 或 dim=-1：第三维（词表维度）

                # 例子1：在最后一维上的softmax
                probs = torch.softmax(logits, dim=-1)  # dim=-1 等价于 dim=2
                '''
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                #使用top-p采样选择下一个token
                next_token = sample_top_p(probs, top_p)
            else:
                '''
                # 如果温度为0，则直接选择概率最大的token
                # 因为温度为0时，logits的值已经代表了概率
                # 所以直接取logits的最后一个位置的最大值
                '''
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # reshape(-1) 将张量展平成一维
            # 例如：
            # 原始形状: [[1], [2]] -> 展平后: [1, 2]
            next_token = next_token.reshape(-1)
            '''
            条件选择器，类似三目运算符
            torch.where(condition, x, y)
            # 如果condition为True，选择x
            # 如果condition为False，选择y

            # 假设我们有：
            input_text_mask = [
                [True,  True,  False],  # 序列1：前2个是真实文本
                [True,  False, False]   # 序列2：前1个是真实文本
            ]

            tokens = [
                [1, 2, 0],  # 序列1
                [3, 0, 0]   # 序列2
            ]

            cur_pos = 1  # 当前处理第2个位置
            next_token = [5, 6]  # 新生成的token

            # torch.where的执行：
            result = torch.where(
                input_text_mask[:, cur_pos],  # [True, False]
                tokens[:, cur_pos],           # [2, 0]
                next_token                    # [5, 6]
            )
            # result = [2, 6]
            # - 序列1：True，保留原token 2
            # - 序列2：False，使用新token 6

            '''
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            # 将生成/保留的token放回tokens张量的对应位置
            tokens[:, cur_pos] = next_token
#如果logprobs为true,那么他会计算每个token的对数概率，并将其存储在token_logprobs中
            if logprobs:
                '''
                # 计算每个token的对数概率
                # 使用交叉熵损失函数，忽略填充token
                # 计算从prev_pos + 1到cur_pos的token的对数概率
                '''
                #我们要计算的是每个输入token预测下一个token的概率，所以要加1，位置1的预测目标是位置2的token
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),                    #将logits的维度从(batch_size, seq_len, vocab_size)转换为(batch_size, vocab_size, seq_len),转置使形状符合cross_entropy要求
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],       #目标token序列
                    reduction="none",                                 #不进行平均或求和，返回每个样本的损失
                    ignore_index=pad_id,                             #忽略填充token的损失
                )
            #如果生成的token是EOS（结束符），则将eos_reached标记为True
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            #更新prev_pos，用于下一次迭代
            prev_pos = cur_pos
            #如果所有样本都到达EOS，则停止生成
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()  #将tensor张量转化为Python列表
        out_tokens, out_logprobs = [], []   #初始化输出token和logprobs列表
        #遍历tokens张量，将每个样本的token和logprobs添加到out_tokens和out_logprobs列表中
        for i, toks in enumerate(tokens.tolist()):
            # 决定了从哪里开始截取最终的输出序列
            # 如果echo为True，则从0开始截取，否则从prompt_tokens[i]的长度开始截取
            start = 0 if echo else len(prompt_tokens[i])
            # 截取最终的输出序列，长度为prompt_tokens[i]的长度加上max_gen_len
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]

            probs = None  #初始化probs为None
            if logprobs:
                #如果logprobs为True，则从token_logprobs中截取最终的输出序列
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            #如果生成的序列中包含EOS（结束符），则截取到EOS的位置
            if self.tokenizer.eos_id in toks:
                #找到EOS在toks中的位置
                eos_idx = toks.index(self.tokenizer.eos_id)
                #截取到EOS的位置
                toks = toks[:eos_idx]
                #如果logprobs为True，则截取到EOS的位置
                probs = probs[:eos_idx] if logprobs else None
            #将截取后的token和probs添加到out_tokens和out_logprobs列表中
            out_tokens.append(toks)
            out_logprobs.append(probs)
        #返回最终的输出token和logprobs
        return (out_tokens, out_logprobs if logprobs else None)

    #text_completion方法用于对提示列表进行文本补全，接收多个参数，用于控制生成过程
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        使用语言生成模型对提示列表进行文本补全。

        参数：
            prompts (List[str])：需要补全的文本提示列表
            temperature (float, optional)：用于控制采样随机性的温度值。默认为 0.6
            top_p (float, optional)：核采样的概率阈值。默认为 0.9
            max_gen_len (Optional[int], optional)：生成补全序列的最大长度
                如果未提供，则设置为模型最大序列长度减 1
            logprobs (bool, optional)：是否计算词元的对数概率。默认为 False
            echo (bool, optional)：是否在生成输出中包含提示词元。默认为 False

        返回：
            List[CompletionPrediction]：补全预测列表，每个包含生成的文本补全

        注意：
            此方法为提供的提示生成文本补全，使用核采样引入可控的随机性。
            如果 logprobs 为 True，将计算每个生成词元的对数概率。
        """
        #如果max_gen_len为None，则设置为模型最大序列长度减1
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        #将输入的文本提示（prompts）转换为词元（token）ID的列表
        #prompts 是一个字符串列表，每个字符串代表一个文本提示。通过列表推导式，逐个处理每个提示。
        #使用分词器将每个提示字符串 x 编码为词元ID列表。
        #bos=True：在编码的词元ID列表前添加一个开始符（BOS, Beginning of Sequence）ID。这通常用于标记序列的开始。
        #eos=False：不在编码的词元ID列表后添加结束符（EOS, End of Sequence）ID。这意味着生成的序列不包含结束标记。
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        #调用generate方法生成文本补全
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        #如果logprobs为True，则返回每个生成的token和对应的log概率
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),   #将整个词元ID序列 t 解码为一个完整的字符串   "你好！"
                    "tokens": [self.tokenizer.decode(x) for x in t],  #逐个将词元ID x 解码为对应的单个词或字符。  ["你", "好", "！"]
                    "logprobs": logprobs_i,  #
                }
                #t 和 logprobs_i 分别代表一个生成的词元ID序列及其对应的对数概率列表
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        #如果logprobs为False，则返回每个生成的token
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    #chat_completion方法用于为对话列表生成助手回复，接收多个参数，用于控制生成过程
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        使用语言生成模型为对话列表生成助手回复。

        参数：
            dialogs (List[Dialog])：对话列表，每个对话是消息列表
            temperature (float, optional)：用于控制采样随机性的温度值。默认为 0.6
            top_p (float, optional)：核采样的概率阈值。默认为 0.9
            max_gen_len (Optional[int], optional)：生成回复序列的最大长度
                如果未提供，则设置为模型最大序列长度减 1
            logprobs (bool, optional)：是否计算词元的对数概率。默认为 False

        返回：
            List[ChatPrediction]：聊天预测列表，每个包含助手生成的回复

        异常：
            AssertionError：如果对话中最后一条消息不是来自用户
            AssertionError：如果对话角色不符合要求的 'user'、'assistant' 和可选的 'system' 顺序

        注意：
            此方法为提供的对话生成助手回复。
            它使用核采样引入文本生成中的可控随机性。
            如果 logprobs 为 True，将计算每个生成词元的对数概率。
        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        #初始化prompt_tokens为空列表
        prompt_tokens = []
        #初始化unsafe_requests为空列表
        unsafe_requests = []
        #遍历对话列表，处理每个对话
        for dialog in dialogs:
            #检查对话中是否包含特殊标签，如果包含，则将unsafe_requests列表中对应位置的元素设置为True
            '''
            . SPECIAL_TAGS：这是一个列表，包含了一些特殊标记。这些标记可能用于格式化或控制对话的行为。
            2. dialog：这是一个对话列表，其中每个元素是一个消息字典，包含 role 和 content 键。
            3. 列表推导式：
            for msg in dialog：遍历对话中的每条消息。
            for tag in SPECIAL_TAGS：遍历所有特殊标记。
            tag in msg["content"]：检查当前消息的内容中是否包含当前特殊标记。
            4. any() 函数：
            any() 函数返回 True 如果列表中有任何一个元素为 True，否则返回 False。
            在这个上下文中，any() 用于检查对话中是否有任何消息包含任何一个特殊标记。
            5. unsafe_requests.append()：
            将 any() 的结果（True 或 False）添加到 unsafe_requests 列表中。
            如果对话中包含特殊标记，则标记为不安全请求。

            例子：
            # 假设有以下特殊标记和对话：
            SPECIAL_TAGS = ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]

            dialog = [
                {"role": "user", "content": "你好，请问今天天气如何？"},
                {"role": "assistant", "content": "[INST] 这是一个测试 [/INST]"}
            ]

            # 检查对话：
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )

            # 结果：
            # 因为第二条消息包含 "[INST]" 和 "[/INST]"，所以 any() 返回 True
            # unsafe_requests = [True]
            '''
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            #如果对话的第一条消息是系统消息，则将对话的角色和内容进行调整
            '''
            # 假设有以下对话：
            dialog = [
                {"role": "system", "content": "这是系统消息"},
                {"role": "user", "content": "用户的提问"},
                {"role": "assistant", "content": "助手的回答"}
            ]

            # 处理系统消息：
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],  # "user"
                        "content": B_SYS
                        + dialog[0]["content"]  # 系统消息内容
                        + E_SYS
                        + dialog[1]["content"],  # 用户的提问
                    }
                ] + dialog[2:]

            # 结果：
            # dialog = [
            #     {
            #         "role": "user",
            #         "content": "<<SYS>>这是系统消息<</SYS>>用户的提问"
            #     },
            #     {"role": "assistant", "content": "助手的回答"}
            # ]
            '''
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]

                # 确保对话中的消息角色按照特定的顺序排列：系统消息（可选），然后是用户消息和助手消息交替出现。
                '''
                assert condition, message
                condition：需要检查的条件表达式。如果为 False，则触发异常。
                message（可选）：当条件为 False 时，抛出的 AssertionError 异常中附带的消息。通常用于解释断言失败的原因。

                dialog[::2]：提取对话中所有 偶数索引（0, 2, 4...） 的消息。
                例如：dialog = [msg0, msg1, msg2, msg3] → 提取 [msg0, msg2]
                这些消息的 role 必须是 "user"。
                dialog[1::2]：提取对话中所有 奇数索引（1, 3, 5...） 的消息。
                例如：dialog = [msg0, msg1, msg2, msg3] → 提取 [msg1, msg3]
                这些消息的 role 必须是 "assistant"。
                
                允许的角色：仅限 "system"、"user"、"assistant"。
                合法顺序：
                可选的 "system" 消息（仅限第一条）。
                之后必须严格交替出现 "user" 和 "assistant"。
                '''
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )

#是将对话（dialog）中的 用户消息（user） 和 助手消息（assistant） 按照特定格式编码为 token 序列，并将所有消息的 token 序列拼接成一个完整的列表。详细见：typora部分解释：（将对话的每个消息编码为token，结果保存在`prompt_tokens`中）部分
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],  # 提取对话中所有 偶数索引（0, 2, 4...） (user)的消息。
                        dialog[1::2],  # 提取对话中所有 奇数索引（1, 3, 5...） (assistant)的消息。
                    )
                ],    #将每个消息的token序列拼接成一个完整的列表
                [],   #初始化dialog_tokens为空列表 ，用于累加所有 token 序列。
            )
            #确保对话的最后一条消息是用户消息   
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            #将最后一条用户消息编码为token，并添加到dialog_tokens中
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            #将dialog_tokens添加到prompt_tokens中
            prompt_tokens.append(dialog_tokens)
        #调用generate函数，生成补全文本的token和对数概率。
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        #如果logprobs为True，则返回每个生成的token和对应的log概率
        if logprobs:
            return [
                {

                    # generation：生成的助手消息，包含角色（role）和内容（content）。
                    # tokens：生成的 token 序列的解码结果（每个 token 对应的文本）。
                    # logprobs：每个 token 的对数概率。
    
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)  #将 token 序列 t 解码为文本。
                        #如果unsafe为True，则返回UNSAFE_ERROR，否则返回解码后的token
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],  #将 token 序列 t 中的每个 token 解码为文本。
                    "logprobs": logprobs_i,
                }
                
                # generation_tokens = [[1, 2, 3], [4, 5, 6]]
                # generation_logprobs = [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]]
                # unsafe_requests = [False, True]
                # for t, logprobs_i, unsafe in zip(generation_tokens, generation_logprobs, unsafe_requests):
                # print(t, logprobs_i, unsafe)
                
                # 输出：
                #     [1, 2, 3] [-0.1, -0.2, -0.3] False
                #     [4, 5, 6] [-0.4, -0.5, -0.6] True
                
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        #如果logprobs为False，则返回每个生成的token
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]



#实现了 Top-p 采样（核采样），用于从概率分布中选择下一个生成的 token。Top-p 采样的核心思想是保留概率最高的 token，直到累积概率超过阈值 p，然后从这些 token 中随机采样。
def sample_top_p(probs, p):
    """
    对概率分布执行 top-p（核）采样。

    参数：
        probs (torch.Tensor)：概率分布张量
        p (float)：top-p 采样的概率阈值

    返回：
        torch.Tensor：采样的词元索引

    注意：
        Top-p 采样选择累积概率质量超过阈值 p 的最小词元集合。
        分布基于选定的词元重新归一化。
    """
    '''
    probs：输入的概率分布，形状为 [batch_size, vocab_size]。
    torch.sort：对概率分布按最后一个维度（dim=-1）排序，返回排序后的概率和对应的索引。
    probs_sort：排序后的概率分布。
    probs_idx：排序后的索引。

    例如： 
    probs = [0.1, 0.4, 0.2, 0.3] (索引：0 1 2 3)
    probs_sort = [0.4, 0.3, 0.2, 0.1]
    probs_idx = [1, 3, 2, 0]  # 原始索引
    probs_sum = [0.4, 0.7, 0.9, 1.0]  # 累积概率
    p = 0.9
    mask = [False, False, False, True]  # 只有最后一个 token 的累积概率超过 0.9
    probs_sort = [0.4, 0.3, 0.2, 0.0]  # 最后一个 token 的概率被置为 0
    probs_sort = [0.4 / 0.9, 0.3 / 0.9, 0.2 / 0.9, 0.0]  # 重新归一化
    probs_idx = [1, 3, 2, 0]
    next_token = 1  # 采样结果为索引 1 即0.3,而0.3的原始索引是3
    next_token = 3  # 映射回原始索引
    '''
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    #计算排序后概率的累积和
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    #创建一个掩码，标记哪些 token 的累积概率超过阈值 p，超过的需要被置零
    mask = probs_sum - probs_sort > p
    #将掩码中标记为True的元素置零
    probs_sort[mask] = 0.0
    #将置零后的概率分布重新归一化
    #probs_sort.sum(dim=-1, keepdim=True)：计算保留 token 的概率总和。
    #div_：将概率分布除以其总和，重新归一化。
    #keepdim=True：保持维度不变，以便后续操作正常运行
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    #torch.multinomial从归一化后的概率分布中随机采样一个 token,num_samples=1：采样一个 token。
    next_token = torch.multinomial(probs_sort, num_samples=1)
    #根据采样结果，从原始索引中获取对应的 token
    #torch.gather：将采样结果映射回原始概率分布中的索引。
    #probs_idx：排序后的索引。
    #next_token：采样结果的索引。
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
