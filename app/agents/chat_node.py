from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage, AIMessage


class ChatSubgraphState(TypedDict, total=False):
    """Chat子图专用状态，字段与母图完全独立"""
    user_input: Optional[str]
    chat_history: List[BaseMessage]
    chat_response: Optional[str]
    chat_meta: Dict[str, Any]


# Chat子图节点函数（纯函数，无需类封装）

async def chat_process(state: ChatSubgraphState) -> Dict[str, Any]:
    """处理普通对话"""
    user_input = state.get("user_input") or ""
    
    # 模拟对话响应（实际应调用LLM）
    response_text = f"收到您的问题：「{user_input}」\n\n这是一个普通对话回复，我可以帮您解答各种问题。"
    
    chat_msg = AIMessage(content=response_text)
    
    return {
        "chat_response": response_text,
        "chat_history": state.get("chat_history", []) + [chat_msg],
        "chat_meta": {"node": "chat_process", "status": "success"},
    }


async def chat_finish(state: ChatSubgraphState) -> Dict[str, Any]:
    """完成节点"""
    return {"chat_meta": {"node": "chat_finish", "status": "complete"}}


def build_chat_subgraph() -> CompiledStateGraph:
    """构建Chat子图（工厂函数）"""
    builder = StateGraph(ChatSubgraphState)

    builder.add_node("chat_process", chat_process)
    builder.add_node("chat_finish", chat_finish)

    builder.add_edge(START, "chat_process")
    builder.add_edge("chat_process", "chat_finish")
    builder.add_edge("chat_finish", END)

    return builder.compile()
