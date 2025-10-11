from typing import List
from pydantic import BaseModel,Field
from app.schemas.schema import Message
from app.graphs.graph_state import AgentState




class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """向记忆库添加单次记忆消息"""
        self.messages.append(message)
        # 添加记忆长短限制
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """向记忆库添加多轮记忆消息"""
        self.messages.extend(messages)
        # 添加记忆长短限制
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """清除所有的信息"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """获得最近N轮消息"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """消息转化为dict"""
        return [msg.to_dict() for msg in self.messages]
