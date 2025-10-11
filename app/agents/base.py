from abc import ABC
from app.logger import logger
from app.cores.llm import LLM
from app.memory.base import Memory
from typing import Optional, Dict, List
from app.graphs.graph_state import AgentState
from app.schemas.schema import ROLE_TYPE, Message
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field, model_validator

class BaseAgent(ABC, BaseModel):
    """
    Agent的抽象基类, 用于设置Agent的属性以及执行
    
    """
    name: str = Field(..., description="Agent名称")
    description: Optional[str] = Field(None, description="Agent描述")

    # 大模型依赖
    llm: LLM = Field(default_factory=LLM, description="大模型类型")
    
    # 记忆库
    memory: Memory = Field(default_factory=Memory, description="Agent的Memory存储")

    # Agent可用的工具
    tool: Dict[str, str] = Field(default_factory=dict, description="Agent可用的工具列表")

    # Prompt提示词
    system_prompt: Optional[str] = Field(None, description="系统级的指示Prompt")
    next_step_prompt: Optional[str] = Field(None, description="下一步动作的Prompt")
    
    # 执行控制
    max_steps: int = Field(default=10, description="最大反复执行次数")
    current_step: int = Field(default=0, description="当前次数")

    # duplicate_threshold: int = 2

   
    class Config:
        arbitrary_types_allowed = True # 允许添加基础类型除外的类型
        extra = "allow"  # 允许添加其他字段 
    

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """以默认参数初始化Agent状态"""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self
    
    def update_agent_memory(self
                            , role: ROLE_TYPE  # type: ignore
                            , content: str
                            , base64_image: Optional[str]=None
                            , **kwargs) -> None:
        
        # FIXME 后续需要修改和增加限制
        """向Agent添加融合后的记忆

        Args:
            role (ROLE_TYPE): 自定义类型
            content (str): 内容
            base64_image (Optional[str], optional): 图像数据，其实这个感觉不是很有用
            kwargs (any): 添加附加的参数，

        Returns:
            _type_: 无需返回
        """
        message_dict = {
            "user": Message.user_message,
            "ai": Message.assistant_message,
            "system": Message.system_message,
            "toolcall": lambda content, **kw : Message.tool_message(content, **kw),
        }

        if role not in message_dict:
            logger.error(f"不支持当前类型角色: {role}")
            raise ValueError(f"不支持当前类型角色: {role}")
        
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_dict[role](content, **kwargs))
                


    @property
    def messages(self) -> List[Message]:
        """获取最近的Agent记忆."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """set最近的Agent记忆."""
        self.memory.messages = value    

    # async def run(self, request: Optional[str] = None) -> str:
    #     """
    #     Execute the agent's main loop asynchronously.

    #         Args:
    #             request: Optional initial user request to process.

    #         Returns:
    #             A string summarizing the execution results.

    #         Raises:
    #         RuntimeError: If the agent is not in IDLE state at start.
    #     """
    #     return ""
    # if self.state != AgentState.IDLE:
    #     raise RuntimeError(f"Cannot run agent from state: {self.state}")

    # if request:
    #     self.update_memory("user", request)

    # results: List[str] = []
    # async with self.state_context(AgentState.RUNNING):
    # while (
    #      self.current_step < self.max_steps and self.state != AgentState.FINISHED
    #         ):
    #     self.current_step += 1
    #     step_result = await self.step()

    #     # Check for stuck state
    #     if self.is_stuck():
    #         self.handle_stuck_state()

    #         results.append(f"Step {self.current_step}: {step_result}")

    #     if self.current_step >= self.max_steps:
    #         self.current_step = 0
    #         self.state = AgentState.IDLE
    #         results.append(f"Terminated: Reached max steps ({self.max_steps})")
    # await SANDBOX_CLIENT.cleanup()
    # return "\n".join(results) if results else "No steps executed"