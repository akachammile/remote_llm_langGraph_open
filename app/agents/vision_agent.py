import re
import json
from app.logger import logger
from langgraph.types import Command
from app.agents.base import BaseAgent
from typing import Dict, Literal, Dict
from app.graphs.graph_state import AgentState
from langgraph.graph import StateGraph, START, END
from app.tools.image_segmentation_tool import ImageSegmentationTool
from app.prompts.vision_prompt  import SYSTEM_PROMPT, NEXT_STEP_PROMPT


class VisionAgent(BaseAgent):
    name: str = "VisionAgent"
    description: str = "用于处理图像相关的具体任务, 当前只能负责图像分割、目标检测、分析图像内容三种具体任务"
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    task_map: Dict[str, str] = {
        "segmentation": "node_segmentation",
        "describe": "node_describe",
        "clarification": END,
    }
    tool: Dict[str, str] = {
        "function_name": "node_segmentation",
        "function_name": "node_describe",
    }
    async def node_describe(self, state: AgentState) -> AgentState:
        """仅分析和描述图像内容。"""
        logger.info("--- VisionAgent: node_describe正在执行图像描述 ---")
        prompt_template = f"""
        你是一个遥感领域的图像分析专家,擅长从遥感的专业角度解读图像内容
        - 描述图像内容：请用中文描述图像所处的场景，其次是图像中的物体，并给出一个完整的描述。
        - 描述图像中的物体： 比如描述图像中的建筑、道路、水、云等
        并详细具体的描述图像的内容并总
        用户问题：{state["question"]}
        """
        response = await self.llm.ask_tool(prompt_template, state)
        state["image_content"] = response.content
        state["messages"].append(f"（内部产出：图像分析结果为 '{response.content[:50]}")
        state["sub_task"] = "分析图像内容并总结任务已完成"
        return state

    def route_next_agent(self, state: dict) -> str:
        # FIXME, 需要额外添加条件
        """判断路由条件，根据节点返回的内容判断下一个需要执行的节点

        Args:
            state (dict): _description_

        Returns:
            str: _description_
        """
        return state.get("next_agent", "exit")
    async def vision_agent_supervisor(self, state: AgentState) -> AgentState: # type: ignore
        """VisionAgent的Supervisor节点, 决定下一个具体任务节点"""
        if not state["image_data"]:
            # return Command(goto=END, update={"messages": "缺少图像数据，请重新确定任务节点或提示用户上传图像", "reflection": True})
            state["messages"] = "缺少图像数据，请重新确定任务节点或提示用户上传图像"
            state["sub_task"] = "缺少图像数据，请结束任务"
            state["reflection"] = True
            state["next_agent"] = "exit"
            return state
        parsed = await self._analyze_intent(state)
        goto = self._map_task_to_node(parsed["next_agent"])
        state["next_agent"] = goto
        return state
    

    def node_segmentation(self, state: AgentState) -> AgentState:# type: ignore
        logger.info("--- VisionAgent: node_segmentation正在执行图像描述 ---")

        image_data = state["image_data"]
        if image_data is not None:
            segmented_bytes, segmentation_info, path = ImageSegmentationTool().segment_image(state["image_data"], state["image_path"])
            state["messages"] = "分割处理完毕"
            state["processed_image_path"] = [path]
            state["sub_task"] = "分割任务处理完毕"
            state["reflection"] = True
            return state
            # return Command(goto=END, update={"messages":"分割处理完毕","processed_image_path": [path], "sub_task":"分割任务处理完毕", "reflection": True})
        return state

    def node_detection(self, state: AgentState) -> Command[Literal[END]]:# type: ignore
        image_data = getattr(state, "image_data", None)
        if image_data is not None:
            segmented_bytes, segmentation_info, path = ImageSegmentationTool().segment_image(state["image_data"])
            state["processed_image_path"] = path
        return state

    async def _analyze_intent(self, state: AgentState) -> AgentState:
        """调用 LLM, 返回 JSON"""
        # FIXME 这里需要重新分析，这里如果没有图像，则需要大模型重新反思决策
        if state.get("sub_task"):
            prompt_template: str = self.next_step_prompt.format(sub_task = state["sub_task"]
                                                                , question = state["question"])
        else:
            prompt_template: str = self.system_prompt.format(question = state["question"]
                                                             , content = state.get("image_content", "无")
                                                             , processed_image_path = state.get("processed_image_path", "无"))

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
            "node_segmentation": "node_segmentation",
            "node_describe": "node_describe",
            "exit": "exit",  # 直接跳到 END
        }
        return task_map.get(task_type, END)
    
    def build_subgraph(self):
        vision_subgraph = StateGraph(AgentState)
        vision_subgraph.add_node("vision_agent_supervisor", self.vision_agent_supervisor)
        vision_subgraph.add_node("node_segmentation", self.node_segmentation)
        vision_subgraph.add_node("node_describe", self.node_describe)

        vision_subgraph.add_edge(START, "vision_agent_supervisor")
        vision_subgraph.add_edge("node_describe", "vision_agent_supervisor")
        vision_subgraph.add_edge("node_segmentation", "vision_agent_supervisor")
        
        vision_subgraph.add_conditional_edges(
            "vision_agent_supervisor",
            self.route_next_agent,
            {
                "node_describe": "node_describe",
                "node_segmentation": "node_segmentation",
                "exit": END  # 当决策为 'exit' 时，退出子图
            }
        )
        vision_subgraph = vision_subgraph.compile()
        return vision_subgraph
