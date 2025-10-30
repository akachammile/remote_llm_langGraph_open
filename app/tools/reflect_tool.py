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

# 描述反映工具的用途
_REFLECT_TOOL = "一个批判性思考和错误分析工具。调用此工具可评估前一个执行步骤的结果，判断是否成功，并根据需要建议修正或下一步行动。"


class ReflectTool(BaseTool):
    """
    反射工具，用于评估前一步骤的结果并进行批判性思考。
    """
    name: str = "reflect_tool"
    description: str = _REFLECT_TOOL

    # 反射工具的输入参数
    parameters: dict = {
        "type": "object",
        "properties": {
            "last_step_action": {
                "type": "string",
                "description": "前一个执行的步骤的简短描述或名称。",
            },
            "last_step_result": {
                "type": "string",
                "description": "前一个步骤的执行结果，包括成功输出、错误信息或中间状态。",
            },
            "original_goal": {
                "type": "string",
                "description": "用户最初的完整目标或问题。",
            }
        },
        # 实际执行时，这些参数应该由系统提供，以驱动LLM进行反射
        "required": ["last_step_action", "last_step_result", "original_goal"],
        "additionalProperties": False,
    }

    async def execute(self, last_step_action: str, last_step_result: str, original_goal: str) -> dict:
        """
        执行反射逻辑。通过 LLM 评估前一步的结果并提出下一步建议。
        
        Args:
            last_step_action: 前一个执行的步骤的描述。
            last_step_result: 前一个步骤的执行结果（JSON字符串或文本）。
            original_goal: 用户最初的目标。
            
        Returns:
            dict: 包含 LLM 反思和下一步建议的结构化结果。
        """
        
        # 1. 构建 LLM 的系统指令 (System Prompt)
        # LLM 的输出结构要求严格的 JSON 格式，包含评估结果和建议。
        REFLECT_PROMPT = f"""
        你是一个批判性反思专家。你的任务是根据给定的目标和前一个步骤的执行结果，评估该步骤是否成功，是否达到了预期效果。
        
        请严格遵循以下要求：
        1. 评估必须是客观且简洁的。
        2. 你的输出必须是严格的 JSON 对象格式。
        
        原始用户目标：{original_goal}
        
        前一个步骤的动作描述：{last_step_action}
        前一个步骤的执行结果：{last_step_result}
        
        JSON Schema 必须遵循以下结构：
        {{
            "success": <boolean, 评估该步骤是否成功完成其目标>,
            "critique": <string, 批判性分析该步骤的结果，指出成功点或失败原因>,
            "suggestion": <string, 基于批判，为下一步行动提供明确的建议。例如：'继续执行下一步规划', '重新规划任务', '重试当前步骤' 等>
        }}
        """
        
        # 2. 调用 LLM 进行反射
        system_msgs = Message.system_message(REFLECT_PROMPT)
        
        # 将用户目标、动作和结果作为当前消息传入，帮助 LLM 进行上下文理解
        # 这里可以直接让 LLM 基于 system_msgs 的上下文进行反射，不需要额外的 user_message
        # 但为了遵循 PlanningTool 的 ask_v2 格式，我们可能需要构造一个 "空" 或 "引导性" 的 user_message
        reflection_message = Message.user_message(f"请对以上步骤进行批判性反思，并输出 JSON 结果。")
        
        try:
            # 假设 self.llm.ask_v2 能够处理 system_msgs 并返回结构化结果
            reflection_results = await self.llm.ask_v2([reflection_message], system_msgs=[system_msgs])
            logger.info(f"Reflection results: {reflection_results}")
            
            # 3. 结果返回
            # 实际应用中，这里需要对 reflection_results (通常是字符串) 进行 JSON 解析和验证
            # return ToolResult(
            #     output={"reflection": reflection_results, "status": "reflected"}, 
            #     tool_name=self.name
            # ).to_dict()

        except Exception as e:
            logger.error(f"Error during reflection: {e}")
            # return ToolFailure(
            #     error=f"Reflection tool failed: {str(e)}",
            #     tool_name=self.name
            # ).to_dict()

# 示例：如果您需要一个完整的反射流程，可以参考PlanningTool中的执行逻辑。