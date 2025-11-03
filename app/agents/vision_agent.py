
from pydantic import Field
from app.logger import logger
from langgraph.types import Command
from app.agents.base import BaseAgent
from typing import Dict, Literal, Dict
from app.graphs.graph_state import AgentState
from langgraph.graph import StateGraph, START, END
from app.tools.image_segmentation_tool import ImageSegmentationTool
from app.prompts.vision_prompt  import SYSTEM_PROMPT, NEXT_STEP_PROMPT
from app.tools.tool_collection import ToolCollection

class VisionAgent(BaseAgent):
    name: str = "VisionAgent"
    description: str | None = "用于处理图像相关的具体任务, 当前只能负责图像分割、目标检测、分析图像内容三种具体任务"
    system_prompt: str | None = SYSTEM_PROMPT
    next_step_prompt: str | None = NEXT_STEP_PROMPT
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            ImageSegmentationTool(),
        )
    )

    
 
    async def vision_agent_supervisor(self, state: AgentState) -> AgentState: # type: ignore
        """VisionAgent的Supervisor节点, 决定下一个具体任务节点"""
        # 保证列表字段与任务初始化状态在子图返回时保持一致
        state["tasks_initialized"] = True
        if not state["image_data"]:
            # 确保 sub_task 已初始化
            if state.get("sub_task") is None:
                state["sub_task"] = []
            state["reflection"] = True
            return state
        task = state["sub_task"].pop(0)
        tool_name = task.get("tool_name")
        if tool_name == "vision_image_segmentation":
            vision_image_segmentation = self.available_tools.get_tool(name=tool_name)
            params = {
                        "image_data": state["image_data"],
                        "image_path": state["image_path"],
                        "state": state,
                    }
            segmented_bytes, segmentation_info, save_path = await vision_image_segmentation.execute(**params)
            state["processed_image_path"].append(save_path)
        return state
    

    def build_subgraph(self):
        vision_subgraph = StateGraph(AgentState)
        vision_subgraph.add_node("vision_agent_supervisor", self.vision_agent_supervisor)
        vision_subgraph.add_edge(START, "vision_agent_supervisor")
        vision_subgraph.add_edge("vision_agent_supervisor", END)  # 直接结束
        return vision_subgraph.compile()