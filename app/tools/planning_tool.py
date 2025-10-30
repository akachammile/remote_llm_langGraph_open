import io
import cv2
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from app.logger import logger
from datetime import datetime
from app.cores.config import config
from app.graphs.graph_state import AgentState
from app.schemas.schema import Message, ToolChoice
from app.tools.base import BaseTool, ToolFailure, ToolResult

_PLANNING_TOOL = "一个高级任务规划器。调用此工具可将一个复杂的用户目标分解为一系列清晰、可执行的子任务和工具调用步骤。"


class PlanningTool(BaseTool):
    name: str = "planning_tool"
    description: str = _PLANNING_TOOL

    parameters: dict = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "用户的问题，此处对用户的问题进行任务分解",
            }
        },
        "required": [],
        "additionalProperties": False,
    }

    async def execute(self, user_message: Message, tools: list[dict]) -> dict:
        plan_results = []
        PLAN_PROMPT = f""""
        你是一个任务分解专家。你的任务是将用户提供的复杂目标分解为一系列按顺序执行的步骤。
        你的输出必须是严格的 JSON 数组格式，其中每个对象代表一个步骤。
        
        可用的原子工具列表及其描述：
        {tools}
        
        JSON Schema 必须遵循以下结构：
        [
        {{
            "step_id": <int>,
            "action": <string, 简短描述该步骤要做什么>,
            "tool_name": <string>,
            "parameters": <dict, 严格符合工具 schema 的参数字典>
        }},
        ...
        ]
        """
        system_msgs = Message.system_message(PLAN_PROMPT)
        plan_results = await self.llm.ask_v2([user_message], system_msgs=[system_msgs])
        logger.info(f"Plan results: {plan_results}")
        # """
        # Args:
        #     tasks: [{"tool": "segmentation_tool", "params": {...}}, ...]
        #     tools: {"segmentation_tool": SegmentationTool实例, ...}
        # Returns:
        #     dict: 子任务执行结果的汇总
        # """
        # results = []
        # for task in tasks:
        #     tool_name = task["tool"]
        #     params = task["params"]
        #     if tool_name not in tools:
        #         results.append({"tool": tool_name, "error": "tool not found"})
        #         continue
        #     tool = tools[tool_name]
        #     try:
        #         output = tool.execute(state=state, **params)
        #         results.append({"tool": tool_name, "output": output})
        #     except Exception as e:
        #         results.append({"tool": tool_name, "error": str(e)})

        # # 可以在这里更新 state 或做全局规划决策
        # state["sub_task"] = "planning completed"
        # return {"results": results, "status": "done"}
