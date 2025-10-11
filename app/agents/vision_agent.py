import re
import json
from langgraph.types import Command
from app.agents.base import BaseAgent
from typing import Dict, Literal, Dict
from app.graphs.graph_state import AgentState
from langgraph.graph import StateGraph, START, END
from app.prompts.vision_prompt  import SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from app.tools.image_segmentation_tool import ImageSegmentationTool


class VisionAgent(BaseAgent):
    name: str = "VisionAgent"
    description: str = "用于处理图像相关的具体任务, 当前只能负责图像分割、目标检测、变化检测三种具体任务"
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = ""
    task_map: Dict[str, str] = {
        "segmentation": "node_segmentation",
        "detection": "node_detection",
        "analysis": "analysis",
        "clarification": END,
    }
    tool: Dict[str, str] = {
        "function_name": "node_segmentation",
        "function_name": "node_detection",
    }


    async def vision_agent_supervisor(self, state: AgentState) -> Command[Literal["node_detection", "node_segmentation", END]]: # type: ignore
        """VisionAgent的Supervisor节点, 决定下一个具体任务节点"""

        if not state["image_data"]:
            return Command(goto=END, update={"messages": "缺少图像数据，请重新确定任务节点或提示用户上传图像", "reflection": True})
        parsed = await self._analyze_intent(state)
        goto = self._map_task_to_node(parsed["task_type"])
        if goto == END:
            return Command(goto="__end__")
        else:
            return Command(goto=goto)
    

    def node_segmentation(self, state: AgentState) -> Command[Literal[END]]:# type: ignore
        image_data = state["image_data"]
        if image_data is not None:
            segmented_bytes, segmentation_info, path = ImageSegmentationTool().segment_image(state["image_data"], state["image_path"])
            return Command(goto=END, update={"messages":"分割处理完毕","processed_image_path": [path]})
        return Command(goto=END)

    def node_detection(self, state: AgentState) -> Command[Literal[END]]:# type: ignore
        image_data = getattr(state, "image_data", None)
        if image_data is not None:
            segmented_bytes, segmentation_info, path = ImageSegmentationTool().segment_image(state["image_data"])
            state["processed_image_path"] = path
        return state

    async def _analyze_intent(self, state: AgentState) -> AgentState:
        """调用 LLM, 返回 JSON"""
        # FIXME 这里需要重新分析，这里如果没有图像，则需要大模型重新反思决策
        prompt_template: str = self.system_prompt.format(question = state["question"])
        response = await self.llm.ask_tool(prompt_template, state)
        json_str = response.content
        match = re.search(r"\{[\s\S]*\}", json_str)

        if match:
            json_str = match.group(0)
        return json.loads(json_str)

    
    def _map_task_to_node(self, task_type: str) -> str:
        """
        根据任务类型返回下一个节点：
        - segmentation/detection/analysis 会走对应节点
        - clarification 直接结束
        """
        task_map = {
            "segmentation": "node_segmentation",
            "detection": "node_detection",
            "analysis": "analysis",
            "clarification": END,  # 直接跳到 END
        }
        return task_map.get(task_type, END)
    
    def build_subgraph(self):
        vision_subgraph = StateGraph(AgentState)
        vision_subgraph.add_node("vision_agent_supervisor", self.vision_agent_supervisor)
        vision_subgraph.add_node("node_segmentation", self.node_segmentation)
        vision_subgraph.add_node("node_detection", self.node_detection)

        vision_subgraph.add_edge(START, "vision_agent_supervisor")
        vision_subgraph.add_edge("vision_agent_supervisor", END)
        vision_subgraph.add_edge("node_detection", END)
        vision_subgraph.add_edge("node_segmentation", END)
        vision_subgraph = vision_subgraph.compile()
        return vision_subgraph
