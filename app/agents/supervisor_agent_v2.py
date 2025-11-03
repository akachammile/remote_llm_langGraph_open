import re
import os
import json
import base64
import traceback
import time
from app.agents.base import BaseAgent
from app.agents.chat_agent import ChatAgent
from app.agents.doc_agent import DocAgent
from app.agents.vision_agent import VisionAgent
from app.logger import logger
from pydantic import BaseModel, Field
from app.database.utils import KnowledgeFile
from app.graphs.graph_state import AgentState
from typing import List, Optional, Dict, Union, Set, Any
from app.schemas.schema import Message, ToolChoice
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from app.prompts.supervisor_prompt import SYSTEM_PROMPT, TOOL_PROMPT, USER_PROMPT
from app.tools.tool_collection import ToolCollection
from app.tools.planning_tool import PlanningTool
from app.tools.file_process_tool import FileProcessTool
from app.tools.image_segmentation_tool import ImageSegmentationTool
from app.database.db.repository.message_repository import (
    add_message_to_db,
    update_message,
    get_message_by_id,
    filter_message,
)


# ============ Schema定义 ============
class SupervisorDecision(BaseModel):
    """Supervisor的规划决策结果"""
    next_agent: str = Field(description="下一个执行的Agent名称")
    reasoning: str = Field(description="决策理由")
    requires_tools: List[str] = Field(default_factory=list, description="需要的工具列表")


class FinalResponse(BaseModel):
    """最终聚合响应"""
    status: str = Field(description="执行状态: success, error, partial")
    answer: str = Field(description="最终答案")
    sources: Dict[str, Any] = Field(default_factory=dict, description="答案来源")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="执行元数据")


class SupervisorAgent(BaseAgent):
    name: str = "SupervisorAgent"
    description: Optional[str] = "用于管理和协调多个子Agent的工作"
    system_prompt: Optional[str] = SYSTEM_PROMPT
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
        # 调用BaseAgent初始化
        super().__init__()  # pyright: ignore
        self._graph: Optional[CompiledStateGraph] = None
        self.agent_infos: List[Dict[str, str]] = self.get_all_agent_info()
        self.placehold_prompt: str = self._build_prompt()

        if not self.placehold_prompt:
            logger.warning(
                "SupervisorAgent 初始化警告：placehold_prompt 为空，使用 fallback_prompt"
            )
            self.placehold_prompt = ""

        self.chat_history: str = ""

    @staticmethod
    def get_all_agent_info() -> List[Dict[str, str]]:
        """获取所有子Agent中的name以及description信息"""

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
            return ""

    async def chat_response(
        self, message: str, file_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:  # type: ignore
        """处理用户输入并调用图执行"""
        IMAGE_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "bmp", "tif"}
        encoded_string = ""
        extension = ""
        image_path = ""
        
        if self._graph is None:
            self._graph = await self.create_supervisor_graph()

        try:
            # 处理文件列表
            if file_list:
                for file_path in file_list:
                    base_path, ext = os.path.splitext(file_path)
                    ext = ext.lower().lstrip(".")
                    if ext in IMAGE_EXTENSIONS:
                        image_path = base_path
                        extension = ext
                        with open(file_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
            # 初始化state
            state: Dict[str, Any] = {  # type: ignore
                "question": message,
                "image_data": encoded_string,
                "image_format": extension,
                "image_path": image_path,
                "memory": self.memory.get_recent_messages(10),
                "sub_task": [],
                "processed_image_path": [],
                "agents_used": [],
                "execution_time": 0,
                "final_response": {},
            }

            start_time = time.time()
            response: Dict[str, Any] = await self._graph.ainvoke(state)  # type: ignore
            response["execution_time"] = time.time() - start_time
            
            return response
        except Exception as e:
            logger.error(f"Error: {e}\n{traceback.format_exc()}")
            raise

    def route_to_agent(self, state: Dict[str, Any]) -> str:
        """根据Supervisor的决策路由到对应的Agent"""
        next_agent = state.get("next_agent", "aggregator")
        
        routing_map = {
            "DocAgent": "doc_agent_node",
            "VisionAgent": "vision_agent_node",
            "ChatAgent": "chat_agent_node",
            "END": "aggregator",
        }
        
        return routing_map.get(next_agent, "aggregator")

    async def top_level_supervisor(self, state: AgentState) -> AgentState:  # type: ignore  # pyright: ignore
        """顶层Supervisor节点 - 允许模型自主上报下一个Agent"""
        logger.info(f"🤔 Supervisor开始规划，问题: {state['question'][:50]}...")
        
        image_data: str = state.get("image_data", "") or ""  # pyright: ignore
        user_message = Message.user_message(
            content=state["question"], 
            base64_image=image_data  # pyright: ignore
        )
        system_message = Message.system_message(self._get_supervisor_system_prompt())
        
        # 调用LLM进行规划 - 使用JSON Schema约束输出
        decision = await self._ask_with_schema(
            messages=[system_message, user_message],
            response_schema=SupervisorDecision
        )
        
        # decision 是 SupervisorDecision 对象或字典
        reasoning = decision.reasoning if isinstance(decision, SupervisorDecision) else decision.get("reasoning", "")  # pyright: ignore
        next_agent = decision.next_agent if isinstance(decision, SupervisorDecision) else decision.get("next_agent", "ChatAgent")  # pyright: ignore
        
        logger.info(f"📋 规划结果: {reasoning}")
        
        state["next_agent"] = next_agent  # pyright: ignore
        state["planning_reasoning"] = reasoning  # pyright: ignore
        state["messages"] = reasoning  # pyright: ignore
        
        return state
    
    def _get_supervisor_system_prompt(self) -> str:
        """返回Superviso的系统提示词

        包含所有可用Agent的描述、例子等
        """
        agent_list = "\n".join(
            [f"- {agent['name']}: {agent['description']}" for agent in self.agent_infos]
        )
        
        return f"""你是一个强大的任务规划专家，有以下自主 Agent 可供你使用：

{agent_list}

需要根据用户的问题，决定最接近的一个Agent来处理。

规则：
- 如林用户上传了图像，优先考虑 VisionAgent
- 如果用户始指文档操作（写入、生成报告等），使用 DocAgent
- 默认使用 ChatAgent 处理漢通对话

输出严格按照以下 JSON 格式：
{{
    "next_agent": "AgentName",
    "reasoning": "为何选择这个Agent的操作"
}}
"""
    
    async def _ask_with_schema(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        response_schema: type,
        max_retries: int = 3,
    ) -> Union[SupervisorDecision, Dict[str, Any]]:  # pyright: ignore
        """
        使用不JSON Schema约束的LLM调用，确保输出符合指定的结构。
        如果输出不符合schema，自动重试
        """
        import json
        
        for attempt in range(max_retries):
            try:
                # 调用LLM
                response = await self.llm.ask_v2(
                    messages=[msg.to_dict() if isinstance(msg, Message) else msg for msg in messages],  # pyright: ignore
                    stream=False
                )
                
                # 尝试提取JSON
                json_match = re.search(r'\{[^{}]*(?:"next_agent"[^{}]*)?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # 没有找JSON，生成默认决策
                    logger.warning(f"第{attempt+1}次尝试：没有找JSON，输出: {response}")
                    if attempt == max_retries - 1:
                        return self._default_routing_decision()
                    messages.append(Message.assistant_message(response))
                    messages.append(Message.user_message(
                        "输出格式不符合要求，请严格按照JSON格式输出：\n{\"next_agent\": \"...\", \"reasoning\": \"...\"}\n输出严格是上JSON，不要有任何其他文字。"
                    ))
                    continue
                
                # 解析JSON
                try:
                    data = json.loads(json_str)
                    
                    # 验证必填字段
                    if "next_agent" not in data or "reasoning" not in data:
                        logger.warning(f"第{attempt+1}次尝试：JSON中丢失必填字段")
                        if attempt == max_retries - 1:
                            return self._default_routing_decision()
                        messages.append(Message.assistant_message(response))
                        messages.append(Message.user_message(
                            "输出的JSON中缺少必填字段: next_agent, reasoning\n输出格式: {\"next_agent\": \"ChatAgent\", \"reasoning\": \"...\"}"
                        ))
                        continue
                    
                    # 验证next_agent是否有效
                    valid_agents = [a["name"] for a in self.agent_infos]
                    if data["next_agent"] not in valid_agents:
                        logger.warning(f"第{attempt+1}次尝试：next_agent不是有效的Agent: {data['next_agent']}, 有效Agent: {valid_agents}")
                        if attempt == max_retries - 1:
                            # 使用最接近的Agent
                            return self._default_routing_decision()
                        messages.append(Message.assistant_message(response))
                        messages.append(Message.user_message(
                            f"你指定的Agent '{data['next_agent']}' 无效。\n有效的Agent只有: {', '.join(valid_agents)}\n仅需选择这些有效Agent之一。"
                        ))
                        continue
                    
                    # 成功
                    logger.info(f"第{attempt+1}次尝试成功: {data}")
                    return SupervisorDecision(**data)  # type: ignore
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"第{attempt+1}次尝试：JSON解析失败: {e}, JSON字符串: {json_str}")
                    if attempt == max_retries - 1:
                        return self._default_routing_decision()
                    messages.append(Message.assistant_message(response))
                    messages.append(Message.user_message(
                        "JSON输出有语法错误，请改正，不要有任何不必要的字符。"
                    ))
                    continue
                    
            except Exception as e:
                logger.error(f"第{attempt+1}次尝试错误: {e}")
                if attempt == max_retries - 1:
                    return self._default_routing_decision()
        
        return self._default_routing_decision()
    
    def _default_routing_decision(self) -> SupervisorDecision:
        """默认路由决策 - 当模型无法正常输出时使用"""
        logger.warning("使用默认路由决策: ChatAgent")
        return SupervisorDecision(
            next_agent="ChatAgent",
            reasoning="由于模型无法正常输出，默认选择ChatAgent"
        )
    
    async def aggregator_node(self, state: AgentState) -> AgentState:  # pyright: ignore
        """聚合所有子Agent的执行结果，生成最终响应"""
        logger.info("📊 聚合层开始收集各Agent结果...")
        
        # 获取消息并转换为字符串
        messages = state.get("messages", "")  # pyright: ignore
        if messages is None:
            messages_str = ""
        elif isinstance(messages, str):
            messages_str = messages
        elif isinstance(messages, list):
            messages_str = str(messages)
        else:
            messages_str = getattr(messages, 'content', str(messages))
        
        # 整理响应
        final_response = FinalResponse(
            status="success",
            answer=messages_str,
            sources={
                "image_sources": state.get("processed_image_path", []),  # pyright: ignore
                "doc_sources": state.get("processed_doc_path", []),  # pyright: ignore
            },
            metadata={
                "agents_used": state.get("agents_used", []),  # pyright: ignore
                "execution_time": state.get("execution_time", 0),  # pyright: ignore
                "sub_tasks": state.get("sub_task", []),  # pyright: ignore
            }
        )
        
        state["final_response"] = final_response.model_dump()  # pyright: ignore
        logger.info(f"✅ 聚合完成，最终答案: {final_response.answer[:100]}...")
        
        return state

    async def create_supervisor_graph(self):
        """构建Supervisor的状态图"""
        if self._graph is None:
            try:
                supervisor_builder = StateGraph(AgentState)

                # 获取各子Agent的子图
                vision_subgraph = VisionAgent().build_subgraph()
                doc_subgraph = DocAgent().build_subgraph()
                chat_subgraph = ChatAgent().build_subgraph()

                # 添加节点
                supervisor_builder.add_node(
                    "supervisor", self.top_level_supervisor
                )
                supervisor_builder.add_node("vision_agent_node", vision_subgraph)
                supervisor_builder.add_node("doc_agent_node", doc_subgraph)
                supervisor_builder.add_node("chat_agent_node", chat_subgraph)
                supervisor_builder.add_node("aggregator", self.aggregator_node)

                # 添加边
                supervisor_builder.add_edge(START, "supervisor")
                
                # 条件路由：根据Supervisor的决策流转到不同的Agent
                supervisor_builder.add_conditional_edges(
                    "supervisor",
                    self.route_to_agent,
                    {
                        "vision_agent_node": "vision_agent_node",
                        "doc_agent_node": "doc_agent_node",
                        "chat_agent_node": "chat_agent_node",
                        "aggregator": "aggregator",
                    },
                )
                
                # 所有子Agent执行完后都流向聚合节点
                supervisor_builder.add_edge("vision_agent_node", "aggregator")
                supervisor_builder.add_edge("doc_agent_node", "aggregator")
                supervisor_builder.add_edge("chat_agent_node", "aggregator")
                
                # 聚合后结束
                supervisor_builder.add_edge("aggregator", END)
                
                self._graph = supervisor_builder.compile()
                logger.info("✅ Supervisor状态图创建成功")
            except Exception as e:
                logger.error(f"❌ Graph创建失败: {str(e)}")
                raise e
        return self._graph
