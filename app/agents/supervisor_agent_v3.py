from langchain_core.messages.base import BaseMessage
import re
import os
import json
import base64
import traceback

from tenacity import retry
from app.agents import *
from app.logger import logger
from pydantic import Field
from app.database.utils import KnowledgeFile
from app.graphs.graph_state import AgentState, AgentStateV2
from langgraph.graph import MessagesState
from langchain_core.tools import tool
from typing import Any, List, Optional, Dict, Union, Set
from app.schemas.schema import Message, ToolChoice
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from app.prompts.supervisor_prompt import SYSTEM_PROMPT, TOOL_PROMPT, USER_PROMPT
from app.tools.tool_collection import ToolCollection
from app.tools.planning_tool import PlanningTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from app.tools.file_process_tool import FileProcessTool
from app.tools.image_segmentation_tool import ImageSegmentationTool,image_segmentation_tool
from app.tools.tools_test import TEST_TOOLS
from app.database.db.repository.message_repository import (
    add_message_to_db,
    update_message,
    get_message_by_id,
    filter_message,
)


class SupervisorAgent(BaseAgent):
    name: str = "SupervisorAgent"
    description: str = "用于管理和协调多个子Agent的工作"
    system_prompt: str = SYSTEM_PROMPT
    tool_prompt: str = TOOL_PROMPT
    user_prompt: str = USER_PROMPT
    current_step: int = 1
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PlanningTool(),
            ImageSegmentationTool(),
            FileProcessTool(),
        )
    )

    def __init__(self):
        super().__init__()
        self._graph: Optional[CompiledStateGraph] = None
        self.agent_infos: List[Dict[str, str]] = self.get_all_agent_info()
        self.placehold_prompt: str = self._build_prompt()
        self.vision_subgraph = VisionAgent().build_subgraph()
        self.doc_subgraph = DocAgent().build_subgraph()
        self.chat_subgraph = ChatAgent().build_subgraph()


        if not self.placehold_prompt:
            logger.warning(
                "SupervisorAgent 初始化警告：placehold_prompt 为空，使用 fallback_prompt"
            )
            self.placehold_prompt = ""

        self.chat_history: str = ""

    @staticmethod
    def get_all_agent_info() -> List[Dict[str, str]]:
        """获取所有子Agent中的name以及description信息"""

        # TODO 此处需要优化，系统Agent的描述只需要更新一次，除非有改动， 还没修改
        def all_subclasses(cls):
            subclasses = set(cls.__subclasses__())
            for subclass in cls.__subclasses__():
                subclasses.update(all_subclasses(subclass))
            return subclasses

        agents_description = []
        for agent_cls in all_subclasses(BaseAgent):
            # 增加所有Agent名称、功能及其工具描述
            if agent_cls.model_fields["name"].default != "SupervisorAgent":
                name = (
                    agent_cls.model_fields["name"].default
                    if "name" in agent_cls.model_fields
                    else agent_cls.__name__
                )
                description = (
                    agent_cls.model_fields["description"].default
                    if "description" in agent_cls.model_fields
                    else ""
                )
                tool = (
                    agent_cls.model_fields["tool"].default
                    if "tool" in agent_cls.model_fields
                    else ""
                )

                agents_description.append(
                    {"name": name, "description": description, "tool": tool}
                )
        return agents_description

    @staticmethod
    def get_agents() -> List[str]:
        """获取所有子Agent的name列表"""
        agent_infos = SupervisorAgent.get_all_agent_info()
        return [a["name"] for a in agent_infos]

    def _build_prompt(self) -> str:
        """封装system_prompt和用户问题为ChatPromptTemplate并格式化为messages"""
        try:
            tool_list = "\n".join(
                [
                    # f'{agent["name"]}: {agent["description"]}'
                    str(
                        {
                            "name": agent["name"],
                            "description": agent["description"],
                            "tool": agent["tool"],
                        }
                    )
                    for agent in self.agent_infos
                    if agent.get("name") != "SupervisorAgent"
                ]
            )
            prompt_template: str = self.tool_prompt.format(
                tool_list=tool_list,
            )
            return prompt_template
        except Exception as e:
            logger.error(f"当前build_prompt_error错误为: {str(traceback.format_exc())}")

    # @tool
    # def chat_node(self, state: str) -> str:
    #     return {"messages": self.llm.ask_v2(messages=[HumanMessage(content=message)])}


    # async def chat_response(self, message: str, file_list: List[Union[str]]) -> AgentState:
    #     # FIXME 此处需要优化支持类型
    #     # result = cut_query(text=message)

    #     IMAGE_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "bmp", "tif"}
    #     encoded_string = ""
    #     extension = ""
    #     image_path = ""
    #     # 创建图
    #     if self._graph is None:
    #         self._graph = await self.create_supervisor_graph()

    #     # 更新记忆库
    #     # if message:
    #     #     self.update_agent_memory(role="user", content=message)
    #     #     logger.info(f"总体记忆:{self.memory.messages}")

    #     try:
    #         if file_list:
    #             for file_path in file_list:
    #                 image_path, extension = os.path.splitext(
    #                     file_path
    #                 )  # 在循环内获取扩展名
    #                 extension = extension.lower().lstrip(".")
    #                 if extension in IMAGE_EXTENSIONS:
    #                     with open(file_path, "rb") as image_file:
    #                         encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    #         state: AgentState = {
    #             # ========== 基础信息 ==========
    #             "name": self.name,
    #             "question": message,
    #             "last_agent": None,
                
    #             # ========== 任务管理 ==========
    #             "next_agent": [],
    #             "sub_task": [],
    #             "history": [],
                
    #             # ========== 工具相关 ==========
    #             "tool_require": None,
    #             "tool_call": None,
                
    #             # ========== 记忆与上下文 ==========
    #             "memory": self.chat_history,
                
    #             # ========== 图像相关 ==========
    #             "image_data": encoded_string,
    #             "image_path": image_path,
    #             "image_format": extension,
    #             "image_content": None,
    #             "image_uri": None,
    #             "initial_image_description": None,
    #             "processed_image_path": [],
                
    #             # ========== 文档相关 ==========
    #             "processed_doc_path": None,
                
    #             # ========== 控制标志 ==========
    #             "reflection": None,
    #             "tasks_initialized": False,
    #             "is_related": False,
                
    #             # ========== 步骤控制 ==========
    #             "max_steps": None,
    #             "repeat_step": None,
                
    #             # ========== 消息列表 ==========
    #             "messages": [],
    #         }
    #         response: AgentState = await self._graph.ainvoke(input=state)
    #         return response
    #     except Exception as e:
    #         logger.error(f"Error: {e}\n{traceback.format_exc()}")



    async def top_level_supervisor(self, state: AgentStateV2):  # type: ignore
        """顶层Supervisor节点, 决定下一个子Agent"""   
        print(state)
        user_message = Message.user_message(content=state["question"], base64_image=state["image_data"])
        system_message = Message.system_message(self.system_prompt)
        # ai_message = Message.assistant_message(content=state["messages"][-1].content)
    
        response = await self.llm.ask_tool_v3(
                messages=[user_message],
                system_msgs=[system_message],
                tools=TEST_TOOLS
            )
        new_messages: list[str | list[str | dict[Any, Any]] | BaseMessage] = state.get("messages", []) + [response.content]

        return {
            "messages":[response] + state["messages"]
        }
     
    async def should_continue(self, state: AgentStateV2):
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]
        tool_calls = last_message.tool_calls
        # If the LLM makes a tool call, then perform an action
        if tool_calls:
            print(f"Tool calls {tool_calls}")
            return "tool_node"

        return END
   
    async def tool_node(self, state: dict):
        """Performs the tool call"""

        result = []
        tools_by_name = {tool.name: tool for tool in TEST_TOOLS}

        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
            

    async def create_supervisor_graph(self):
        """构建Supervisor的状态图"""
        if self._graph is None:
            try:
                supervisor_builder = StateGraph(AgentStateV2)

                
                # 添加节点
                supervisor_builder.add_node("top_level_supervisor", self.top_level_supervisor)
                supervisor_builder.add_node("tool_node", self.tool_node)
                # supervisor_builder.add_node("VisionAgent", self.vision_subgraph)
                # supervisor_builder.add_node("DocAgent", self.doc_subgraph)
                # supervisor_builder.add_node("ChatAgent", self.doc_subgraph)

                # 添加边
                supervisor_builder.add_edge(START, "top_level_supervisor")

                supervisor_builder.add_conditional_edges(
                    "top_level_supervisor",
                    self.should_continue,
                    ["tool_node", END]
                )
                supervisor_builder.add_edge("tool_node", "top_level_supervisor")
            
                
               
                self._graph = supervisor_builder.compile()
                logger.info("Supervisor状态图创建成功")
            except Exception as e:
                logger.error("Graph创建失败", error=str(e))

                raise e
        return self._graph
