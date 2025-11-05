from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage, AIMessage


class VisionSubgraphState(TypedDict, total=False):
    """Vision子图专用状态，字段与母图完全独立"""
    input_query: Optional[str]
    input_image_base64: Optional[str]
    input_image_file: Optional[str]
    internal_messages: List[BaseMessage]
    segmentation_result: Optional[Dict[str, Any]]
    output_image_files: List[str]
    output_description: Optional[str]
    execution_meta: Dict[str, Any]


# Vision子图节点函数（纯函数，无需类封装）

async def vision_analyze(state: VisionSubgraphState) -> Dict[str, Any]:
    """分析图像需求"""
    query = state.get("input_query") or ""
    img_b64 = state.get("input_image_base64")
    img_file = state.get("input_image_file")
    
    analysis_msg = AIMessage(
        content=f"Vision子图启动 | 查询: {query[:30]}... | 图像来源: {'base64' if img_b64 else 'file' if img_file else 'none'}"
    )
    
    return {
        "internal_messages": state.get("internal_messages", []) + [analysis_msg],
        "execution_meta": {"node": "vision_analyze", "status": "success"},
    }


async def vision_segment(state: VisionSubgraphState) -> Dict[str, Any]:
    """执行图像分割（模拟）"""
    img_file = state.get("input_image_file") or "unknown.png"
    
    seg_result = {
        "regions": [
            {"label": "object_1", "bbox": [10, 20, 100, 150], "confidence": 0.92},
            {"label": "object_2", "bbox": [200, 50, 280, 180], "confidence": 0.85},
        ],
        "total_regions": 2,
    }
    
    output_files = [
        f"vision_output/segmented_{img_file.split('/')[-1]}",
        f"vision_output/overlay_{img_file.split('/')[-1]}",
    ]
    
    seg_msg = AIMessage(content=f"分割完成：检测到 {seg_result['total_regions']} 个区域")
    
    return {
        "segmentation_result": seg_result,
        "output_image_files": output_files,
        "internal_messages": state.get("internal_messages", []) + [seg_msg],
        "execution_meta": {"node": "vision_segment", "status": "success"},
    }


async def vision_describe(state: VisionSubgraphState) -> Dict[str, Any]:
    """生成描述"""
    seg = state.get("segmentation_result") or {}
    total = seg.get("total_regions", 0)
    
    desc = f"图像分析完成：识别到 {total} 个主要区域，已生成分割掩码与叠加图像。"
    desc_msg = AIMessage(content=desc)
    
    return {
        "output_description": desc,
        "internal_messages": state.get("internal_messages", []) + [desc_msg],
        "execution_meta": {"node": "vision_describe", "status": "success"},
    }


async def vision_finish(state: VisionSubgraphState) -> Dict[str, Any]:
    """完成节点"""
    return {"execution_meta": {"node": "vision_finish", "status": "complete"}}


def build_vision_subgraph() -> CompiledStateGraph:
    """构建Vision子图（工厂函数）"""
    builder = StateGraph(VisionSubgraphState)

    builder.add_node("vision_analyze", vision_analyze)
    builder.add_node("vision_segment", vision_segment)
    builder.add_node("vision_describe", vision_describe)
    builder.add_node("vision_finish", vision_finish)

    builder.add_edge(START, "vision_analyze")
    builder.add_edge("vision_analyze", "vision_segment")
    builder.add_edge("vision_segment", "vision_describe")
    builder.add_edge("vision_describe", "vision_finish")
    builder.add_edge("vision_finish", END)

    return builder.compile()
