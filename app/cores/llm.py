import math
from app.logger import logger
from langchain_ollama import ChatOllama
from app.graphs.graph_state import AgentState
from app.cores.config import LLMSettings, config
from langchain_core.messages import HumanMessage
from typing import Dict, Any, Optional, Union, List
from langchain_core.messages import BaseMessage, AIMessage
from app.tools.custom_decorator_tool import retry_decorator
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSerializable
REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = ["qwen2.5vl:7b"]


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
    def __new__(cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]
    
    # LLM属性初始化
    def __init__(self, config_name: str = "default", llm_config: Optional[LLMSettings] = None):
        if not hasattr(self, "client"):  # 没有llm实例则初始化一个即可
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url
            self.top_k = llm_config.top_k
            self.top_p = llm_config.top_p

            # 设置token计数器
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            # 初始化分词器，考虑使用Qwen的
            try:
                pass
                self.tokenizer = None
            except KeyError:
                pass
                # If the model is not in tiktoken's presets, use cl100k_base as default
                # self.tokenizer = tiktoken.get_encoding("cl100k_base")

            if self.api_type == "ollama":
                self.client = ChatOllama(
                    model = self.model,
                    base_url=self.base_url,
                    # api_key=self.api_key,
                    # temperature=self.temperature,
                    # top_k=self.top_k,
                    # top_p=self.top_p
                )
            # elif self.api_type == "aws":
            #     self.client = BedrockClient()
            # else:
            #     self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

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
    
    
    @retry_decorator
    async def ask_with_images(
        # self,
        # messages: List[Union[dict, Message]],
        # images: List[Union[str, dict]],
        # system_msgs: Optional[List[Union[dict, Message]]] = None,
        # stream: bool = False,
        # temperature: Optional[float] = None,
    ) -> str:
        pass
    

    # @staticmethod
    # def format_messages(
    #     messages: List[Union[dict, BaseMessage]], supports_images: bool = False
    # ) -> List[dict]:
    #     """
    #     Format messages for LLM by converting them to OpenAI message format.

    #     Args:
    #         messages: List of messages that can be either dict or Message objects
    #         supports_images: Flag indicating if the target model supports image inputs

    #     Returns:
    #         List[dict]: List of formatted messages in OpenAI format

    #     Raises:
    #         ValueError: If messages are invalid or missing required fields
    #         TypeError: If unsupported message types are provided

    #     Examples:
    #         >>> msgs = [
    #         ...     Message.system_message("You are a helpful assistant"),
    #         ...     {"role": "user", "content": "Hello"},
    #         ...     Message.user_message("How are you?")
    #         ... ]
    #         >>> formatted = LLM.format_messages(msgs)
    #     """
    #     formatted_messages = []

    #     for message in messages:
    #         # Convert Message objects to dictionaries
    #         if isinstance(message, Message):
    #             message = message.to_dict()

    #         if isinstance(message, dict):
    #             # If message is a dict, ensure it has required fields
    #             if "role" not in message:
    #                 raise ValueError("Message dict must contain 'role' field")

    #             # Process base64 images if present and model supports images
    #             if supports_images and message.get("base64_image"):
    #                 # Initialize or convert content to appropriate format
    #                 if not message.get("content"):
    #                     message["content"] = []
    #                 elif isinstance(message["content"], str):
    #                     message["content"] = [
    #                         {"type": "text", "text": message["content"]}
    #                     ]
    #                 elif isinstance(message["content"], list):
    #                     # Convert string items to proper text objects
    #                     message["content"] = [
    #                         (
    #                             {"type": "text", "text": item}
    #                             if isinstance(item, str)
    #                             else item
    #                         )
    #                         for item in message["content"]
    #                     ]

    #                 # Add the image to content
    #                 message["content"].append(
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {
    #                             "url": f"data:image/jpeg;base64,{message['base64_image']}"
    #                         },
    #                     }
    #                 )

    #                 # Remove the base64_image field
    #                 del message["base64_image"]
    #             # If model doesn't support images but message has base64_image, handle gracefully
    #             elif not supports_images and message.get("base64_image"):
    #                 # Just remove the base64_image field and keep the text content
    #                 del message["base64_image"]

    #             if "content" in message or "tool_calls" in message:
    #                 formatted_messages.append(message)
    #             # else: do not include the message
    #         else:
    #             raise TypeError(f"Unsupported message type: {type(message)}")

    #     # Validate all messages have required fields
    #     for msg in formatted_messages:
    #         if msg["role"] not in ROLE_VALUES:
    #             raise ValueError(f"Invalid role: {msg['role']}")

    #     return formatted_messages
    # @staticmethod
    # def format_messages(
    #     messages: List[Union[dict, Message]], supports_images: bool = False
    # ) -> List[dict]:
    #     """
    #     Format messages for LLM by converting them to OpenAI message format.

    #     Args:
    #         messages: List of messages that can be either dict or Message objects
    #         supports_images: Flag indicating if the target model supports image inputs

    #     Returns:
    #         List[dict]: List of formatted messages in OpenAI format

    #     Raises:
    #         ValueError: If messages are invalid or missing required fields
    #         TypeError: If unsupported message types are provided

    #     Examples:
    #         >>> msgs = [
    #         ...     Message.system_message("You are a helpful assistant"),
    #         ...     {"role": "user", "content": "Hello"},
    #         ...     Message.user_message("How are you?")
    #         ... ]
    #         >>> formatted = LLM.format_messages(msgs)
    #     """
    #     formatted_messages = []

    #     for message in messages:
    #         # Convert Message objects to dictionaries
    #         if isinstance(message, Message):
    #             message = message.to_dict()

    #         if isinstance(message, dict):
    #             # If message is a dict, ensure it has required fields
    #             if "role" not in message:
    #                 raise ValueError("Message dict must contain 'role' field")

    #             # Process base64 images if present and model supports images
    #             if supports_images and message.get("base64_image"):
    #                 # Initialize or convert content to appropriate format
    #                 if not message.get("content"):
    #                     message["content"] = []
    #                 elif isinstance(message["content"], str):
    #                     message["content"] = [
    #                         {"type": "text", "text": message["content"]}
    #                     ]
    #                 elif isinstance(message["content"], list):
    #                     # Convert string items to proper text objects
    #                     message["content"] = [
    #                         (
    #                             {"type": "text", "text": item}
    #                             if isinstance(item, str)
    #                             else item
    #                         )
    #                         for item in message["content"]
    #                     ]

    #                 # Add the image to content
    #                 message["content"].append(
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {
    #                             "url": f"data:image/jpeg;base64,{message['base64_image']}"
    #                         },
    #                     }
    #                 )

    #                 # Remove the base64_image field
    #                 del message["base64_image"]
    #             # If model doesn't support images but message has base64_image, handle gracefully
    #             elif not supports_images and message.get("base64_image"):
    #                 # Just remove the base64_image field and keep the text content
    #                 del message["base64_image"]

    #             if "content" in message or "tool_calls" in message:
    #                 formatted_messages.append(message)
    #             # else: do not include the message
    #         else:
    #             raise TypeError(f"Unsupported message type: {type(message)}")

    #     # Validate all messages have required fields
    #     for msg in formatted_messages:
    #         if msg["role"] not in ROLE_VALUES:
    #             raise ValueError(f"Invalid role: {msg['role']}")

    #     return formatted_messages
    

    # @retry(
    #     wait=wait_random_exponential(min=1, max=60),
    #     stop=stop_after_attempt(6),
    #     retry=retry_if_exception_type(
    #         (Exception, ValueError)
    #     ),  # Don't retry TokenLimitExceeded
    # )
    # async def ask(
    #     self,
    #     messages: List[Union[dict, Message]],
    #     system_msgs: Optional[List[Union[dict, Message]]] = None,
    #     stream: bool = True,
    #     temperature: Optional[float] = None,
    # ) -> str:
    #     """
    #     根据问题进行回答.

    #     Args:
    #         messages: 多轮对话List
    #         system_msgs: 系统消息，可选
    #         stream (bool): 是否流式返回
    #         temperature (float): 采样温度

    #     Returns:
    #         str: 流式/非流式返回content

    #     Raises:
    #         TokenLimitExceeded: Token数量返回上限
    #     """
    #     try:
    #         # 是否支持图像，一般是支持吗，后续是否会支持
    #         supports_images = self.model in MULTIMODAL_MODELS

    #         # 格式化消息
    #         if system_msgs:
    #             system_msgs = self.format_messages(system_msgs, supports_images)
    #             messages = system_msgs + self.format_messages(messages, supports_images)
    #         else:
    #             messages = self.format_messages(messages, supports_images)

    #         # 计算输入Token
    #         input_tokens = self.count_message_tokens(messages)

    #         # 检测是否超过token上限
    #         if not self.check_token_limit(input_tokens):
    #             error_message = self.get_limit_error_message(input_tokens)
    #             # Raise a special exception that won't be retried
    #             raise TokenLimitExceeded(error_message)

    #         params = {
    #             "model": self.model,
    #             "messages": messages,
    #         }

    #         if self.model in REASONING_MODELS:
    #             params["max_completion_tokens"] = self.max_tokens
    #         else:
    #             params["max_tokens"] = self.max_tokens
    #             params["temperature"] = (
    #                 temperature if temperature is not None else self.temperature
    #             )

    #         if not stream:
    #             # Non-streaming request
    #             response:BaseMessage = await self.client.invoke(
    #                 **params, stream=False
    #             )

    #             if not response.content or not response.response_metadata:
    #                 raise ValueError("无效回答")

    #             # # Update token counts
    #             # self.update_token_count(
    #             #     response.usage.prompt_tokens, response.usage.completion_tokens
    #             # )

    #             # return response.choices[0].message.content
    #             return response.content

    #         # Streaming request, For streaming, update estimated token count before making the request
    #         # self.update_token_count(input_tokens)

    #         # response = await self.client.chat.completions.create(**params, stream=True)
    #         response:BaseMessage = await self.client.invoke(**params, stream=True)

    #         collected_messages = []
    #         completion_text = ""
    #         async for chunk in response.content:
    #             chunk_message = chunk or ""
    #             collected_messages.append(chunk_message)
    #             completion_text += chunk_message
    #             print(chunk_message, end="", flush=True)

    #         print()  # Newline after streaming
    #         full_response = "".join(collected_messages).strip()
    #         if not full_response:
    #             raise ValueError("Empty response from streaming LLM")

    #         # estimate completion tokens for streaming response
    #         completion_tokens = self.count_tokens(completion_text)
    #         logger.info(
    #             f"Estimated completion tokens for streaming response: {completion_tokens}"
    #         )
    #         self.total_completion_tokens += completion_tokens

    #         return full_response

    #     # except TokenLimitExceeded:
    #     #     # Re-raise token limit errors without logging
    #     #     raise
    #     # except ValueError:
    #     #     logger.exception(f"Validation error")
    #     #     raise
    #     # except OpenAIError as oe:
    #     #     logger.exception(f"OpenAI API error")
    #     #     if isinstance(oe, AuthenticationError):
    #     #         logger.error("Authentication failed. Check API key.")
    #     #     elif isinstance(oe, RateLimitError):
    #     #         logger.error("Rate limit exceeded. Consider increasing retry attempts.")
    #     #     elif isinstance(oe, APIError):
    #     #         logger.error(f"API error: {oe}")
    #     #     raise
    #     except Exception:
    #         logger.exception(f"Unexpected error in ask")
    #         raise

    # @retry(
    #     wait=wait_random_exponential(min=1, max=60),
    #     stop=stop_after_attempt(6),
    #     retry=retry_if_exception_type(
    #         (Exception, ValueError)
    #     ),  # Don't retry TokenLimitExceeded
    # )
    # async def ask_with_images(
    #     self,
    #     messages: List[Union[dict, Message]],
    #     images: List[Union[str, dict]],
    #     system_msgs: Optional[List[Union[dict, Message]]] = None,
    #     stream: bool = False,
    #     temperature: Optional[float] = None,
    # ) -> str:
    #     """
    #     多模态问答接口，支持文本和图像输入.

    #     Args:
    #         messages: 问题或者其他List
    #         images: 图像的List，可以是base64字符串或者URL
    #         system_msgs: Optional system messages to prepend
    #         stream (bool): Whether to stream the response
    #         temperature (float): Sampling temperature for the response

    #     """
    #     try:
    #         if self.model not in MULTIMODAL_MODELS:
    #             raise ValueError(
    #                 f"Model {self.model} 当前模型不支持图像. 请使用以下模型 {MULTIMODAL_MODELS}"
    #             )

    #         # Format messages with image support
    #         formatted_messages = self.format_messages(messages, supports_images=True)

    #         # Ensure the last message is from the user to attach images
    #         if not formatted_messages or formatted_messages[-1]["role"] != "user":
    #             raise ValueError(
    #                 "The last message must be from the user to attach images"
    #             )

    #         # Process the last user message to include images
    #         last_message = formatted_messages[-1]

    #         # Convert content to multimodal format if needed
    #         content = last_message["content"]
    #         multimodal_content = (
    #             [{"type": "text", "text": content}]
    #             if isinstance(content, str)
    #             else content if isinstance(content, list) else []
    #         )

    #         # Add images to content
    #         for image in images:
    #             if isinstance(image, str):
    #                 multimodal_content.append(
    #                     {"type": "image_url", "image_url": {"url": image}}
    #                 )
    #             elif isinstance(image, dict) and "url" in image:
    #                 multimodal_content.append({"type": "image_url", "image_url": image})
    #             elif isinstance(image, dict) and "image_url" in image:
    #                 multimodal_content.append(image)
    #             else:
    #                 raise ValueError(f"Unsupported image format: {image}")

    #         # Update the message with multimodal content
    #         last_message["content"] = multimodal_content

    #         # Add system messages if provided
    #         if system_msgs:
    #             all_messages = (
    #                 self.format_messages(system_msgs, supports_images=True)
    #                 + formatted_messages
    #             )
    #         else:
    #             all_messages = formatted_messages

    #         # Calculate tokens and check limits
    #         input_tokens = self.count_message_tokens(all_messages)
    #         if not self.check_token_limit(input_tokens):
    #             raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

    #         # Set up API parameters
    #         params = {
    #             "model": self.model,
    #             "messages": all_messages,
    #             "stream": stream,
    #         }

    #         # Add model-specific parameters
    #         if self.model in REASONING_MODELS:
    #             params["max_completion_tokens"] = self.max_tokens
    #         else:
    #             params["max_tokens"] = self.max_tokens
    #             params["temperature"] = (
    #                 temperature if temperature is not None else self.temperature
    #             )

    #         # Handle non-streaming request
    #         if not stream:
    #             response = await self.client.chat.completions.create(**params)

    #             if not response.choices or not response.choices[0].message.content:
    #                 raise ValueError("Empty or invalid response from LLM")

    #             self.update_token_count(response.usage.prompt_tokens)
    #             return response.choices[0].message.content

    #         # Handle streaming request
    #         self.update_token_count(input_tokens)
    #         response = await self.client.chat.completions.create(**params)

    #         collected_messages = []
    #         async for chunk in response:
    #             chunk_message = chunk.choices[0].delta.content or ""
    #             collected_messages.append(chunk_message)
    #             print(chunk_message, end="", flush=True)

    #         print()  # Newline after streaming
    #         full_response = "".join(collected_messages).strip()

    #         if not full_response:
    #             raise ValueError("Empty response from streaming LLM")

    #         return full_response

    #     except TokenLimitExceeded:
    #         raise
    #     except ValueError as ve:
    #         logger.error(f"Validation error in ask_with_images: {ve}")
    #         raise
    #     except OpenAIError as oe:
    #         logger.error(f"OpenAI API error: {oe}")
    #         if isinstance(oe, AuthenticationError):
    #             logger.error("Authentication failed. Check API key.")
    #         elif isinstance(oe, RateLimitError):
    #             logger.error("Rate limit exceeded. Consider increasing retry attempts.")
    #         elif isinstance(oe, APIError):
    #             logger.error(f"API error: {oe}")
    #         raise
    #     except Exception as e:
    #         logger.error(f"Unexpected error in ask_with_images: {e}")
    #         raise

    # @retry(
    #     wait=wait_random_exponential(min=1, max=60),
    #     stop=stop_after_attempt(6),
    #     retry=retry_if_exception_type(
    #         (Exception, ValueError)
    #     ),  # Don't retry TokenLimitExceeded
    # )
    # async def ask_tool(
    #     self,
    #     messages: List[Union[dict, Message]],
    #     system_msgs: Optional[List[Union[dict, Message]]] = None,
    #     timeout: int = 300,
    #     tools: Optional[List[dict]] = None,
    #     tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
    #     temperature: Optional[float] = None,
    #     **kwargs,
    # ) -> BaseMessage | None:
    #     """
    #     Ask LLM using functions/tools and return the response.

    #     Args:
    #         messages: List of conversation messages
    #         system_msgs: Optional system messages to prepend
    #         timeout: Request timeout in seconds
    #         tools: List of tools to use
    #         tool_choice: Tool choice strategy
    #         temperature: Sampling temperature for the response
    #         **kwargs: Additional completion arguments

    #     Returns:
    #         ChatCompletionMessage: The model's response

    #     Raises:
    #         TokenLimitExceeded: If token limits are exceeded
    #         ValueError: If tools, tool_choice, or messages are invalid
    #         OpenAIError: If API call fails after retries
    #         Exception: For unexpected errors
    #     """
    #     try:
    #         # Validate tool_choice
    #         if tool_choice not in TOOL_CHOICE_VALUES:
    #             raise ValueError(f"Invalid tool_choice: {tool_choice}")

    #         # Check if the model supports images
    #         supports_images = self.model in MULTIMODAL_MODELS

    #         # Format messages
    #         if system_msgs:
    #             system_msgs = self.format_messages(system_msgs, supports_images)
    #             messages = system_msgs + self.format_messages(messages, supports_images)
    #         else:
    #             messages = self.format_messages(messages, supports_images)

    #         # Calculate input token count
    #         input_tokens = self.count_message_tokens(messages)

    #         # If there are tools, calculate token count for tool descriptions
    #         tools_tokens = 0
    #         if tools:
    #             for tool in tools:
    #                 tools_tokens += self.count_tokens(str(tool))

    #         input_tokens += tools_tokens

    #         # Check if token limits are exceeded
    #         if not self.check_token_limit(input_tokens):
    #             error_message = self.get_limit_error_message(input_tokens)
    #             # Raise a special exception that won't be retried
    #             raise TokenLimitExceeded(error_message)

    #         # Validate tools if provided
    #         if tools:
    #             for tool in tools:
    #                 if not isinstance(tool, dict) or "type" not in tool:
    #                     raise ValueError("Each tool must be a dict with 'type' field")

    #         # Set up the completion request
    #         params = {
    #             "model": self.model,
    #             "messages": messages,
    #             "tools": tools,
    #             "tool_choice": tool_choice,
    #             "timeout": timeout,
    #             **kwargs,
    #         }

    #         # TODO 暂时不要
    #         # if self.model in REASONING_MODELS:
    #         #     params["max_completion_tokens"] = self.max_tokens
    #         # else:
    #         #     params["max_tokens"] = self.max_tokens
    #         #     params["temperature"] = (
    #         #         temperature if temperature is not None else self.temperature
    #         #     )

    #         params["stream"] = False  # Always use non-streaming for tool requests
    #         response: BaseMessage = await self.client.invoke(
    #             **params
    #         )

    #         # Check if response is valid
    #         if not response.content or not response:
    #             print(response)
    #             # raise ValueError("Invalid or empty response from LLM")
    #             return None

    #         # 更新token使用统计
    #         self.update_token_count(response.content, response.response_metadata)

    #         return response.choices[0].message

    #     except TokenLimitExceeded:
    #         # Re-raise token limit errors without logging
    #         raise
    #     # except ValueError as ve:
    #     #     logger.error(f"Validation error in ask_tool: {ve}")
    #     #     raise
    #     # except OpenAIError as oe:
    #     #     logger.error(f"OpenAI API error: {oe}")
    #     #     if isinstance(oe, AuthenticationError):
    #     #         logger.error("Authentication failed. Check API key.")
    #     #     elif isinstance(oe, RateLimitError):
    #     #         logger.error("Rate limit exceeded. Consider increasing retry attempts.")
    #     #     elif isinstance(oe, APIError):
    #     #         logger.error(f"API error: {oe}")
    #     #     raise
    #     except Exception as e:
    #         logger.error(f"Unexpected error in ask_tool: {e}")
    #         raise

    
    # @retry_decorator(tries=3)
    # async def generate(self,
    #     messages: List[Union[dict]],
    #     system_msgs: Optional[List[Union[dict]]] = None,
    #     stream: bool = True,
    #     temperature: Optional[float] = None,):
    #     pass


    


    
   