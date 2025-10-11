from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from app.agents.base import BaseAgent


class BaseFlow(BaseModel, ABC):
    """workflow基类"""
    agents: Dict[str, BaseAgent]
    tools: Optional[List] = None
    primary_agent_key: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        if isinstance(agents, BaseAgent):
            agents_dict = {"default": agents}
        elif isinstance(agents, list):
            agents_dict = {f"agent_{i}": agent for i, agent in enumerate(agents)}
        else:
            agents_dict = agents

        primary_key = data.get("primary_agent_key")
        if not primary_key and agents_dict:
            primary_key = next(iter(agents_dict))
            data["primary_agent_key"] = primary_key

        data["agents"] = agents_dict

        super().__init__(**data)

    @property
    def primary_agent(self) -> Optional[BaseAgent]:
        return self.agents.get(self.primary_agent_key)

    def get_agent(self, key: str) -> Optional[BaseAgent]:
        return self.agents.get(key)

    def add_agent(self, key: str, agent: BaseAgent) -> None:
        self.agents[key] = agent

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """基于输入执行工作流"""
