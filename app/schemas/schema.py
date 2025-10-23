from enum import Enum
from pydantic import BaseModel, Field, SerializeAsAny
from typing import Dict, List, Optional, Any, Union, Literal
from langgraph.graph import END, StateGraph, START, MessagesState

from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    OPENAI = auto()
    OPENAI_COMPATIBLE = auto()
    AZURE_OPENAI = auto()
    DEEPSEEK = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    VERTEXAI = auto()
    GROQ = auto()
    AWS = auto()
    OLLAMA = auto()
    OPENROUTER = auto()
    FAKE = auto()



class DeepseekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing"""

    DEEPSEEK_CHAT = "deepseek-chat"


class OllamaModelName(StrEnum):
    """https://ollama.com/search"""

    OLLAMA_GENERIC = "ollama"



AllModelEnum: TypeAlias = (
    DeepseekModelName
    | OllamaModelName
)


class AgentState(str, Enum):
    """Agent执行状态枚举类"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    NEXT = ""



class Role(str, Enum):
    """角色消息类型"""
    SYSTEM = "system" #参考Langchain的
    USER = "user"
    ASSISTANT = "assistant"
    AI = "ai"
    TOOL = "tool"
    
ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class ToolChoice(str, Enum):
    """工具选择枚举"""

    NONE = "none" #无需工具
    AUTO = "auto" #自主选择
    REQUIRED = "required" #必须带工具


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore



class Function(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    """消息中的函数类型"""
    id: str
    type: str = "function"
    function: Function

class Message(BaseModel):
    """对话中的消息参数, 其实可使用LangGraph代替"""

    role: ROLE_TYPE = Field(description="对话角色:如系统,模型,工具等")  # type: ignore
    content: Optional[str] = Field(description="对话内容", default=None)
    name: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(description="调用的工具函数", default=None)

    tool_call_id: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(description="附带的图像", default=None)

    run_id: str | None = Field(
        description="对话id",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="相应的元数据",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="自定义的data",
        default={},
    )

    def pretty_repr(self) -> str:
        """使用Pretty方式输出."""
        base_title = self.role.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"当前的两个类型不支持拼接: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """将消息类型转化为字段形式"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.model_dump() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[Union[str, bytes]] = None
    ) -> "Message":
        """创建用户消息,如果带有图像"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """创建系统消息模板"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, base64_image: Optional[str] = None
    ) -> "Message":
        """创建AI消息模板"""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str, base64_image: Optional[str] = None
    ) -> "Message":
        """创建Tool调用消息"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image)

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ):
        """创建ToolCallMessage.

        Args:
            tool_calls: 模型输出的需要调用的函数
            content: Optional message content
            base64_image: Optional base64 encoded image
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )

class AgentInfo(BaseModel):
    """Agent的相关信息."""

    key: str = Field(
        description="agent的key.",
        examples=["research-assistant"],
    )
    description: str = Field(
        description="Agent的描述信息.",
        examples=["意图识别助手"],
    )



class UserInput(BaseModel):
    """Agent的基础输入."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    model: SerializeAsAny[AllModelEnum] | None = Field(
        title="模型",
        description="驱动模型的Agent.",
        default=OllamaModelName.OLLAMA_GENERIC,
        examples=[OllamaModelName.OLLAMA_GENERIC],
    )
    thread_id: str | None = Field(
        description="多轮对话的id,并非线程",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="用户ID,用于持久化对话用户对话.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_config: dict[str, Any] = Field(
        description="需额外传输给Agent的参数Additional configuration to pass through to the agent",
        default={},
        examples=[{"spicy_level": 0.8}],
    )


