
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

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # 可选字段
    logprobs: List[float]  # 可选字段


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
#当用户输入时，包含这些特殊标记产生的错误
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,    #为什么可以默认为空，是因为下面的并行初始化的判断吗？
        seed: int = 1,
    ) -> "Llama":
        """
        通过初始化和加载预训练模型来构建 Llama 实例。

        参数：
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
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")   #NCCL (NVIDIA Collective Communications Library) 后端初始化分布式进程组（用于高性能集群的通信库，加快分布式训练的计算速度
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed 必须在所有进程中保持一致
        torch.manual_seed(seed)
        #why:在分布式训练中，local_rank=0 通常被设计为主进程(约定俗称的)
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")
#加载模型的检查点文件和参数配置
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
#完成模型的最终初始化和加载过程，并打印加载模型花费的时间
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        #模型参数中的词汇量大小
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    #是 PyTorch 提供的一个装饰器（decorator），用于在模型推理（inference）时优化计算和内存使用。它的作用是确保在推理过程中不会进行不必要的计算（如梯度计算），从而提高推理效率并减少内存占用。
    @torch.inference_mode()
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
        #基础配置
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        #当total_len=params.max_seq_len的时候，同时输入又达到了total_len
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                 #将logits的维度从(batch_size, seq_len, vocab_size)转换为(batch_size, vocab_size, seq_len),转置使形状符合cross_entropy要求
                input=logits.transpose(1, 2),  
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

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
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

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
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
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
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]


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
    例子：
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
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
