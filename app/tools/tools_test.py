# ... existing code ...
from langchain_core.tools.base import BaseTool
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from app.cores.llm import LLM



@tool("chat_tool")
def chat_tool(message: str) -> str:
    """一个聊天助手，精通遥感方面的知识，同时支持对图像的解读/内容解析对话以及日常的对话
    参数:
      - message: 用户输入内容问题
    返回:
      - 字符串回应
    """
    return f"你说：{message}"

@tool("image_segmentation_tool")
def image_segmentation_tool(
    image_b64: Optional[str] = None,
    image_path: Optional[str] = None,
    target: Optional[str] = None
) -> Dict[str, Any]:
    """
    测试用图像分割工具。输入base64或路径，返回模拟分割结果。
    参数:
      - image_b64: 图片的base64字符串
      - image_path: 图片文件路径
      - target: 可选的目标类别提示
    返回:
      - 包含分割区域的字典（模拟结果）
    """
    source = "base64" if image_b64 else ("path" if image_path else "none")
    regions = [
        {"label": target or "object", "bbox": [12, 18, 120, 160], "confidence": 0.87},
        {"label": target or "object", "bbox": [220, 80, 300, 200], "confidence": 0.76},
    ]
    return {"source": source, "count": len(regions), "regions": regions}

@tool("doc_generate_tool")
def doc_generate_tool(
    title: str,
    headings: Optional[List[str]] = None,
    bullets_per_heading: int = 3
) -> str:
    """
    文档生成测试工具。根据标题与小节自动生成条目。
    参数:
      - title: 文档标题
      - headings: 小节名称列表
      - bullets_per_heading: 每个小节生成条目数
    返回:
      - 文本内容（markdown风格）
    """
    headings = headings or ["背景", "目标", "方案", "结论"]
    lines = [f"# {title}"]
    for h in headings:
        lines.append(f"## {h}")
        for i in range(1, bullets_per_heading + 1):
            lines.append(f"- {h}要点 {i}")
    return "\n".join(lines)

# @tool("planning_tool_test")
# def planning_tool_test(question: str) -> List[Dict[str, Any]]:
#     """
#     简单任务规划测试工具。把问题拆为线性步骤。
#     参数:
#       - question: 用户问题或目标
#     返回:
#       - 规划步骤的列表（JSON对象）
#     """
#     return [
#         {"step_id": 1, "action": "澄清需求", "tool_name": "chat_tool", "parent": "SupervisorAgent"},
#         {"step_id": 2, "action": "信息检索/生成文档骨架", "tool_name": "doc_generate_tool", "parent": "DocAgent"},
#         {"step_id": 3, "action": "如有图片则进行分割", "tool_name": "image_segmentation_tool", "parent": "VisionAgent"},
#         {"step_id": 4, "action": "汇总与答复", "tool_name": "chat_tool", "parent": "ChatAgent"},
#     ]

TEST_TOOLS: list[BaseTool] = [chat_tool, image_segmentation_tool, doc_generate_tool]
# ... existing code ...