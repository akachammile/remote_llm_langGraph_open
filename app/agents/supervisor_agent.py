import re
import os
import json
import base64
import traceback
from app.agents import *
from app.logger import logger
from pydantic import Field
from app.database.utils import KnowledgeFile
from app.graphs.graph_state import AgentState
from typing import List, Optional, Dict, Union, Set
from app.schemas.schema import Message, ToolChoice
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from app.prompts.supervisor_prompt import SYSTEM_PROMPT, TOOL_PROMPT, USER_PROMPT
from app.tools.tool_collection import ToolCollection
from app.tools.planning_tool import PlanningTool

from app.tools.file_process_tool import FileProcessTool
from app.tools.image_segmentation_tool import ImageSegmentationTool



class SupervisorAgent(BaseAgent):
    name: str = "SupervisorAgent"
    description: str = "用于管理和协调多个子Agent的工作"
    system_prompt: str = SYSTEM_PROMPT
    tool_prompt: str = TOOL_PROMPT
    user_prompt: str = USER_PROMPT
    current_step: int = 1
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PlanningTool(),ImageSegmentationTool(), FileProcessTool(), 
        )
    )

    def __init__(self):
        super().__init__()
        self._graph: Optional[CompiledStateGraph] = None
        self.agent_infos: List[Dict[str, str]] = self.get_all_agent_info()
        self.placehold_prompt: str = self._build_prompt()
        
        if not self.placehold_prompt:
            logger.warning("SupervisorAgent 初始化警告：placehold_prompt 为空，使用 fallback_prompt")
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

    async def chat_response(
        self, message: str, file_list: List[Union[str]]
    ) -> AgentState:
        # FIXME 此处需要优化支持类型
        # result = cut_query(text=message)

        IMAGE_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "bmp", "tif"}
        encoded_string = ""
        extension = ""
        image_path = ""
        # 创建图
        if self._graph is None:
            self._graph = await self.create_supervisor_graph()

        # 更新记忆库
        # if message:
        #     self.update_agent_memory(role="user", content=message)
        #     logger.info(f"总体记忆:{self.memory.messages}")

        try:
            if file_list:
                for file_path in file_list:
                    image_path, extension = os.path.splitext(
                        file_path
                    )  # 在循环内获取扩展名
                    extension = extension.lower().lstrip(".")
                    if extension in IMAGE_EXTENSIONS:
                        with open(file_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

            state: AgentState = {
                "question": message,
                "memory": self.chat_history,
                "image_data": encoded_string,
                "image_format": extension,
                "memory": self.memory.get_recent_messages(10),
                "image_path": image_path,
            }

            response: AgentState = await self._graph.ainvoke(state)
            return response
        except Exception as e:
            logger.error(f"Error: {e}\n{traceback.format_exc()}")


    def route_next_agent(self, state: dict) -> str:
        # FIXME, 需要额外添加条件
        """判断路由条件，根据节点返回的内容判断下一个需要执行的节点

        Args:
            state (dict): _description_

        Returns:
            str: _description_
        """
        return state.get("next_agent", "END")
    
        

    async def top_level_supervisor(self, state: AgentState) -> AgentState:  # type: ignore
        """顶层Supervisor节点, 决定下一个子Agent"""
        # TODO: 需要添加循环判断，比如字节点任务完成后，若是返回Supervisor节点时候，需要增加额外的过滤条件，不可以一直循环的走Supervisor
        # state.setdefault("history", [])
        # state.setdefault("sub_task", [])
        # state["history"].append(state.get("next_agent", "SupervisorAgent"))
        
        # if state.get("sub_task"):
        #     logger.info("🤔 正在思考下一步任务")
        #     state = await self.next_step(state)
        #     return state
        
        user_message = Message.user_message(content=state["question"], base64_image=state["image_data"])
        system_message = Message.system_message(self.system_prompt)
        logger.info(f"🤔 工具情况: {self.available_tools.to_params()}")
        response = await self.llm.ask_tool_v2(
            messages=[user_message],
            system_msgs=[system_message],
            tools=self.available_tools.to_params(),
            tool_choice=ToolChoice.AUTO,
        )
        logger.info(f"🤔 思考结果为: {response}")
        logger.info(f"🔧 调用工具为: {response.tool_calls}")
        if response.tool_calls:
            for tool in response.tool_calls:
                tool_name = tool.function.name if tool.function.name else None
                if tool_name == "image_segmentation":
                    segmentation_tool = self.available_tools.get_tool("image_segmentation")
                    params = {
                        "image_data": state["image_data"],
                        "image_path": state["image_path"],
                        "state":state
                        }
                    segmented_bytes, segmentation_info, save_path = segmentation_tool.execute(**params)
                    state["sub_task"].append(save_path)
                    logger.info(f"图片分割结果为: {save_path}")
                    logger.info(f"state: {state["sub_task"]}")
                elif tool_name == "planning_tool":
                    planning_tool = self.available_tools.get_tool(tool_name)
                    params = {
                        "user_message": user_message,
                        "tools": self.available_tools.to_params()[1:],
                        }
                    _ = await planning_tool.execute(**params)
                    
                    

        
                    
                    
        
        
        # 🧱 构建输入消息（仅首次或重新规划时使用）
        # if not state.get("messages"):  # 如果没有现成上下文，则用基础 prompt
        #     user_message = self.user_prompt.format(query=state["question"])
        #     messages = self.system_prompt + self.placehold_prompt + user_message
        # else:
        #     # 否则保留已有上下文，让模型接着上次状态思考
        #     messages = self.system_prompt + self.placehold_prompt + \
        #         str(state["messages"][-1].content)
        # # response = await self.llm.ask_tool(messages, state)
        # response = await self.llm.ask_tool_v2(messages, state)
        # json_str = response.content
        # match = re.search(r"\{[\s\S]*\}", json_str)
        # if match:
        #     json_str = match.group(0)
        # response = json.loads(json_str)
        # new_state = state.copy()
        
        # if response["task"]["name"]  == "ChatAgent":
        # # 如果是简单对话，直接更新 messages 并准备结束
        #     new_state["next_agent"] = "exit"      # 下一步直接结束            new_state["messages"] = description  # 将回答放入 messages
        #     new_state["messages"] = response["response"]  # 将回答放入 messages
        # else:
        #     # 如果是工具调用，正常分发
        #     new_state["next_agent"] = response["task"]["name"] 
        #     new_state["messages"] = response["response"]
        # return new_state
    # async def chat_node(self, state: AgentState) -> AgentState:
    #     """专门处理简单对话的节点，然后直接结束流程。"""
    #     logger.info("💬 正在处理简单对话，流程即将结束。")
    #     return state
    async def next_step(self, state: AgentState) -> AgentState:
        """LLM-based反思机制"""
        next_step_prompt = f"""
        修正角色定位,你是一个流程计划专家,你需要根据历史任务确定下一个任务。
        - 如果用户问题已经得到完整解答，子任务均标记为“已完成”，你必须结束任务（如结束任务exit）
        根据以上信息结合模板中的文件,规划下一步（如结束任务END或选择其他工具继续执行）。
        输出规划请严格遵循输出要求
        用户问题中的任务是: {state['question']}
        子任务的完成情况为：{state['sub_task']}
        """
        reflect_prompt = self.system_prompt + next_step_prompt + self.placehold_prompt 
        response = await self.llm.ask_tool(reflect_prompt, state)
        match = re.search(r"\{[\s\S]*\}", response.content)
        if match:
            plan = json.loads(match.group(0))
            state["next_agent"] = plan["task"]["name"]
            state["messages"] = plan["response"]
            state["status"] = "replanned"
        else: 
            state["status"] = "failed"
        return state
    
    
    
    async def reflect_and_replan(self, state: AgentState) -> AgentState:
        """LLM-based反思机制"""
        reflect_prompt = f"""
        你是一个任务反思专家,你需要根据历史任务以及提供的相关信息回答是符合当前问题。
        用户问题是: {state['question']}
        子任务的结果为：{state['messages'][-1].content}

        根据以上信息结合模板中的文件,规划下一步（如输出END以提示用户补充信息）。
        输出规划请严格遵循输出要求
        """
        print(f"提示出现的问题是：{state['messages'][-1].content}")
        
        reflect_prompt = self.system_prompt + reflect_prompt + self.placehold_prompt 
        response = await self.llm.ask_tool(reflect_prompt, state)
        match = re.search(r"\{[\s\S]*\}", response.content)
        if match:
            plan = json.loads(match.group(0))
            state["next_agent"] = plan["task"]["name"]
            state["messages"] = plan["response"]
            state["status"] = "replanned"
            state["reflection"] = False
        else:
            state["status"] = "failed"
        return state
    async def create_supervisor_graph(self):
        """构建Supervisor的状态图"""
        if self._graph is None:
            try:
                supervisor_builder = StateGraph(AgentState)
                           
                vision_subgraph = VisionAgent().build_subgraph()
                doc_subgraph = DocAgent().build_subgraph()

                # 添加节点
                supervisor_builder.add_node("top_level_supervisor", self.top_level_supervisor)
                supervisor_builder.add_node("VisionAgent", vision_subgraph)
                # supervisor_builder.add_node("ChatAgent", self.chat_node)
                supervisor_builder.add_node("DocAgent", doc_subgraph)
                # supervisor_builder.add_node("reflection_node", self.reflect_and_replan)
                # supervisor_builder.add_node("planning_node", self.next_step)

                # 添加边
                supervisor_builder.add_edge(START, "top_level_supervisor")
                supervisor_builder.add_conditional_edges(
                    "top_level_supervisor",
                    self.route_next_agent,
                    {
                        "VisionAgent": "VisionAgent",
                        # "ChatAgent": "ChatAgent",
                        "DocAgent": "DocAgent",
                        # "reflection_node": "reflection_node",
                        # "planning_node": "planning_node",
                        "exit": END,
                    },
                )
                supervisor_builder.add_edge("VisionAgent", "top_level_supervisor")
                # supervisor_builder.add_edge("ChatAgent", "top_level_supervisor")
                supervisor_builder.add_edge("DocAgent", "top_level_supervisor")
                self._graph = supervisor_builder.compile()
                logger.info("Supervisor状态图创建成功")
            except Exception as e:
                logger.error("Graph创建失败", error=str(e))

                raise e
        return self._graph
