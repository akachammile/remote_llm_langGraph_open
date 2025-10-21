import math
from app.logger import logger
from langchain_ollama import ChatOllama
from app.graphs.graph_state import AgentState
from app.cores.config import config
from langchain_core.messages import HumanMessage
from typing import Dict, Any, Optional, Union, List
from langchain_core.messages import BaseMessage, AIMessage
from app.tools.custom_decorator_tool import retry_decorator
from langchain_core.vectorstores import InMemoryVectorStore
from openai import APIError, AsyncAzureOpenAI,AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSerializable
from app.schemas.schema import Message, ToolChoice, TOOL_CHOICE_VALUES, ROLE_VALUES, TOOL_CHOICE_TYPE

REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = ["qwen2.5vl:7b", "qwen2.5vl:32b", "z-uo/qwen2.5vl_tools:7b"]


class TokenCounter:
    """Token计数器"""

    # Token常量
    BASE_MESSAGE_TOKENS = 4 
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # 图像大小常量
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """计算输入文本的token数量"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        根据维度信息等计算图像的token

        低细节: 固定85token
        高细节:
        1. 整体裁剪至2048
        2. 短边裁剪至于768
        3. 将缩放后的图像按 512×512 像素块切分，每块大概170token
        4. 在所有块 token 总数上，额外加 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # 低画质
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # 中高画质
        if detail == "high" or detail == "medium":
            # 如果提供了图像尺寸
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """根据图像尺寸计算Token"""
        # 第一步：固定图像尺寸
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # 第二步：图像scale到768
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # 第三步: scale到512
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # 第四步：计算tokens数量
        return (total_tiles * self.HIGH_DETAIL_TILE_TOKENS) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """计算Message的token数量"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """计算一次ToolCall消耗的Token"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """计算Messagelist消耗的Token数量"""
        total_tokens = self.FORMAT_TOKENS  # 固定Token数

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # 消息token数

            # 计算Role token
            tokens += self.count_text(message.get("role", ""))

            # 计算content消耗的token
            if "content" in message:
                tokens += self.count_content(message["content"])

            # 计算tool call消耗的token
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # 计算name和tool_call_id的token
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens
        
        # 返回总token数量
        return total_tokens

class LLM:
    """设置LLM属性以及方法"""
    _instances: Dict[str, "LLM"] = {}

    # 单例模式
    def __new__(cls, config_name: str = "default"):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__()
            cls._instances[config_name] = instance
        return cls._instances[config_name]
    
    # LLM属性初始化
    def __init__(self):
        if not hasattr(self, "client"):  # 没有llm实例则初始化一个即可
            self.base_url = config.MODEL_BASE_URL.get_secret_value()
            self.model = config.MODEL_NAME
            self.max_tokens = config.MODEL_MAX_TOKENS
            self.temperature = config.MODEL_TEMPERATURE
            self.api_type = config.MODEL_API_TYPE
            self.api_key = config.MODEL_API_KEY
            # self.api_version = llm_config.api_version
            self.top_k = config.MODEL_TOP_K
            self.top_p = config.MODEL_TOP_P

            # 设置token计数器
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = config.MODEL_MAX_TOKENS
      

            # 初始化分词器，考虑使用Qwen的
            try:
                pass
                self.tokenizer = None
            except KeyError:
                pass
    

            if self.api_type == "ollama":
                self.client = ChatOllama(
                    model = self.model,
                    base_url=self.base_url,
                    # api_key=self.api_key,
                    # temperature=self.temperature,
                    # top_k=self.top_k,
                    # top_p=self.top_p
                )
          
            elif self.api_type == "openai":
                self.client = AsyncOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )

            self.token_counter = TokenCounter(self.tokenizer)


    def count_tokens(self, text: str) -> int:
        """计算文本的Token数量"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    

    def count_message_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的Token数量"""
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """更新Token使用统计"""
        # 当token的数量达到上限时才统计
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """检查是否超过Token限制"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # 若没设置最大Token限制，则不检查直接返回True
        return True
    
    def get_limit_error_message(self, input_tokens: int) -> str:
        """当Token超过上限生成错误信息"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"
    

    def wrap_model(self, model: BaseChatModel | Runnable[LanguageModelInput, Any], system_prompt: str, state: AgentState) -> RunnableSerializable[AgentState, Any]:
        # FIXME 这里需要重新封装代码，以支持遥感方向的图像解答
        try:
            if state["image_data"]:
                message = HumanMessage(
                        content=[
                            {"type": "text", "text": system_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{state["image_data"]}"},
                            },
                        ],
                    )
            else:
                message = HumanMessage(
                        content=[
                            {"type": "text", "text": system_prompt},
                        ],
                    )

            preprocessor = RunnableLambda(lambda state: [message], name="StateModifier")
        except Exception as e:
            logger.error(f"wrap_model组装失败:{e}")
        return preprocessor | model


    @retry_decorator 
    async def ask_tool(self, prompt:str, state: AgentState)-> AIMessage:
        """询问需要调用的工具"""
        model_runnable = self.wrap_model(self.client, system_prompt=prompt, state=state)
        response = await model_runnable.ainvoke(state)
        return response
    
    
    @retry_decorator
    async def ask_tool_v2(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        使用函数/工具向 LLM 提问并返回响应。

        参数:
        messages: 对话消息列表
        system_msgs: 可选的系统消息，将被添加在前面
        timeout: 请求超时时间（秒）
        tools: 可使用的工具列表
        tool_choice: 工具选择策略
        temperature: 响应采样温度
        **kwargs: 其他补全参数

        返回:
        ChatCompletionMessage: 模型生成的响应

        异常:
        TokenLimitExceeded: 如果超过令牌限制
        ValueError: 如果 tools、tool_choice 或 messages 无效
        OpenAIError: 如果 API 调用在重试后失败
        Exception: 其他意外错误
        """
        try:

            # 验证工具选择
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            # 判断是否支持多模态
            supports_images = self.model in MULTIMODAL_MODELS

            # 格式化消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # 计算输入的token数
            # input_tokens = self.count_message_tokens(messages)

            # 如果需要工具，计算工具描述的token数          
            # tools_tokens = 0
            # if tools:
            #     for tool in tools:
            #         tools_tokens += self.count_tokens(str(tool))

            # input_tokens += tools_tokens

            # # Check if token limits are exceeded
            # if not self.check_token_limit(input_tokens):
            #     error_message = self.get_limit_error_message(input_tokens)
            #     # Raise a special exception that won't be retried
            #     raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            # 验证工具是否有效
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("每个工具必须具备变量名Each tool must be a dict with 'type' field")

            # 设置完成请求
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )
            params["stream"] = False
            
            # Always use non-streaming for tool requests
            # 工具调用使用非流式接口
            response: ChatCompletion = await self.client.chat.completions.create(
                **params
            )

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print(response)
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # Update token counts
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            return response.choices[0].message

        # except TokenLimitExceeded:
        #     # Re-raise token limit errors without logging
        #     raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        # except OpenAIError as oe:
        #     logger.error(f"OpenAI API error: {oe}")
        #     if isinstance(oe, AuthenticationError):
        #         logger.error("Authentication failed. Check API key.")
        #     elif isinstance(oe, RateLimitError):
        #         logger.error("Rate limit exceeded. Consider increasing retry attempts.")
        #     elif isinstance(oe, APIError):
        #         logger.error(f"API error: {oe}")
        #     raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise
    

    @retry_decorator
    async def ask(self, prompt:str, state: AgentState)-> BaseMessage:
        """
        Description: 
            根据问题进行回答,返回BaseMessage

        Args:
        """
        try:
            model_runnable = self.wrap_model(self.client, system_prompt=prompt, state=state)
            response = await model_runnable.ainvoke(state)
        except Exception as e:
            logger.error(f"调用出现意外：{e}")
        return response
    
    
    @retry_decorator
    async def ask_v2(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        向 LLM 发送提示并获取响应。

        参数:
        messages: 对话消息列表
        system_msgs: 可选的系统消息，将被添加在前面
        stream (bool): 是否以流式方式返回响应
        temperature (float): 响应采样温度

        返回:
        str: 生成的响应文本

        异常:
        TokenLimitExceeded: 如果超过令牌限制
        ValueError: 如果消息无效或响应为空
        OpenAIError: 如果 API 调用在重试后失败
        Exception: 其他意外错误
        """
        try:
            # 判断模型是否是多模态模型
            supports_images = self.model in MULTIMODAL_MODELS

            # 判断是否支持多模态
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # # Calculate input token count
            # input_tokens = self.count_message_tokens(messages)

            # # Check if token limits are exceeded
            # if not self.check_token_limit(input_tokens):
            #     error_message = self.get_limit_error_message(input_tokens)
            #     # Raise a special exception that won't be retried
            #     raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            if not stream:
                # 是否流式输出
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("模型输出无效")

                # Update token counts
                # self.update_token_count(
                #     response.usage.prompt_tokens, response.usage.completion_tokens
                # )

                return response.choices[0].message.content

            # self.update_token_count(input_tokens)

            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True)

            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("输出无效")

            # estimate completion tokens for streaming response
            # completion_tokens = self.count_tokens(completion_text)
            # logger.info(
            #     f"Estimated completion tokens for streaming response: {completion_tokens}"
            # )
            # self.total_completion_tokens += completion_tokens

            return full_response

        # except TokenLimitExceeded:
        #     # Re-raise token limit errors without logging
        #     raise
        # except ValueError:
        #     logger.exception(f"Validation error")
        #     raise
        # except OpenAIError as oe:
        #     logger.exception(f"OpenAI API error")
        #     if isinstance(oe, AuthenticationError):
        #         logger.error("Authentication failed. Check API key.")
        #     elif isinstance(oe, RateLimitError):
        #         logger.error("Rate limit exceeded. Consider increasing retry attempts.")
        #     elif isinstance(oe, APIError):
        #         logger.error(f"API error: {oe}")
        #     raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise
        
        
        
    @retry_decorator
    async def ask_stream(self, prompt:str, state: AgentState)-> BaseMessage:
        """
        Description: 
            根据问题进行回答,返回BaseMessage

        Args:
        """
        try:
            model_runnable = self.wrap_model(self.client, system_prompt=prompt, state=state)
            response = await model_runnable.astream(state)
        except Exception as e:
            logger.error(f"调用出现意外：{e}")
        return response
    
    
    

    @staticmethod
    def format_messages(
        messages: List[Union[dict, BaseMessage]], supports_images: bool = False
    ) -> List[dict]:
        """
        将消息格式化为符合 OpenAI 接口规范的消息格式。

        参数说明:
            messages: 消息列表，可以是字典或 Message 对象。
            supports_images: 标志位，指示目标模型是否支持图像输入。

        返回值:
            List[dict]: 以 OpenAI 格式组织的消息列表。

        异常:
            ValueError: 当消息无效或缺少必要字段时抛出。
            TypeError: 当提供了不支持的消息类型时抛出。

        示例:
            >>> msgs = [Message.system_message("你是一个乐于助人的助手"),
            ...     {"role": "user", "content": "你好"},
            ...     Message.user_message("你最近怎么样？")]
            ...        
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # 将Message对象转换为字典
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # 如果message是字典，则确保它包含必要的字段
                if "role" not in message:
                    raise ValueError("Message词典中缺少必填字段'role'")

                # 如果模型支持图像，则处理base64图片
                if supports_images and message.get("base64_image"):
                    # 将内容初始化或转换为适当的格式
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # 将字符串项转换为文本对象
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # 向content中添加图片
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{message['base64_image']}"},
                        }
                    )
                    # 删除base64_image字段，目的是减少占用
                    del message["base64_image"]
                    
                # 如果模型不支持图片，则删除base64图片
                elif not supports_images and message.get("base64_image"):
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                # raise TypeError(f"Unsupported message type: {type(message)}")
                raise TypeError(f"不支持的消息类型: {type(message)}")

        # 验证所有消息都有必需的字段
        # for msg in formatted_messages:
        #     if msg["role"] not in ROLE_VALUES:
        #         raise ValueError(f"无效角色: {msg['role']}")

        return formatted_messages


 
    
