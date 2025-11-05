from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage, AIMessage


class DocSubgraphState(TypedDict, total=False):
    """Doc子图专用状态，字段与母图完全独立"""
    input_user_query: Optional[str]
    input_doc_files: List[str]
    input_context: Optional[str]
    doc_messages: List[BaseMessage]
    retrieval_results: Optional[List[Dict[str, Any]]]
    generated_content: Optional[str]
    output_doc_files: List[str]
    output_summary: Optional[str]
    doc_meta: Dict[str, Any]


# Doc子图节点函数（纯函数，无需类封装）

async def doc_load(state: DocSubgraphState) -> Dict[str, Any]:
    """加载文档并解析"""
    query = state.get("input_user_query") or ""
    files = state.get("input_doc_files", [])
    
    load_msg = AIMessage(
        content=f"Doc子图启动 | 查询: {query[:30]}... | 文档数: {len(files)}"
    )
    
    return {
        "doc_messages": state.get("doc_messages", []) + [load_msg],
        "doc_meta": {"node": "doc_load", "status": "success"},
    }


async def doc_retrieve(state: DocSubgraphState) -> Dict[str, Any]:
    """检索相关文档片段"""
    query = state.get("input_user_query") or ""
    
    # 模拟检索结果
    retrieval = [
        {"chunk_id": "chunk_1", "content": "相关内容片段1", "score": 0.89},
        {"chunk_id": "chunk_2", "content": "相关内容片段2", "score": 0.76},
        {"chunk_id": "chunk_3", "content": "相关内容片段3", "score": 0.68},
    ]
    
    retrieve_msg = AIMessage(content=f"检索完成：找到 {len(retrieval)} 个相关片段")
    
    return {
        "retrieval_results": retrieval,
        "doc_messages": state.get("doc_messages", []) + [retrieve_msg],
        "doc_meta": {"node": "doc_retrieve", "status": "success"},
    }


async def doc_generate(state: DocSubgraphState) -> Dict[str, Any]:
    """基于检索结果生成文档内容"""
    retrieval = state.get("retrieval_results") or []
    query = state.get("input_user_query") or ""
    
    # 模拟文档生成
    generated = f"基于检索结果生成的文档内容：\n\n针对问题「{query}」，从{len(retrieval)}个片段中整合如下信息...\n"
    
    output_files = ["doc_output/generated_report.docx"]
    
    gen_msg = AIMessage(content="文档生成完成")
    
    return {
        "generated_content": generated,
        "output_doc_files": output_files,
        "doc_messages": state.get("doc_messages", []) + [gen_msg],
        "doc_meta": {"node": "doc_generate", "status": "success"},
    }


async def doc_summarize(state: DocSubgraphState) -> Dict[str, Any]:
    """生成摘要"""
    generated = state.get("generated_content") or ""
    output_files = state.get("output_doc_files", [])
    
    summary = f"文档处理完成：生成了 {len(output_files)} 个文档文件，内容长度约 {len(generated)} 字符。"
    summary_msg = AIMessage(content=summary)
    
    return {
        "output_summary": summary,
        "doc_messages": state.get("doc_messages", []) + [summary_msg],
        "doc_meta": {"node": "doc_summarize", "status": "success"},
    }


async def doc_finish(state: DocSubgraphState) -> Dict[str, Any]:
    """完成节点"""
    return {"doc_meta": {"node": "doc_finish", "status": "complete"}}


def build_doc_subgraph() -> CompiledStateGraph:
    """构建Doc子图（工厂函数）"""
    builder = StateGraph(DocSubgraphState)

    builder.add_node("doc_load", doc_load)
    builder.add_node("doc_retrieve", doc_retrieve)
    builder.add_node("doc_generate", doc_generate)
    builder.add_node("doc_summarize", doc_summarize)
    builder.add_node("doc_finish", doc_finish)

    builder.add_edge(START, "doc_load")
    builder.add_edge("doc_load", "doc_retrieve")
    builder.add_edge("doc_retrieve", "doc_generate")
    builder.add_edge("doc_generate", "doc_summarize")
    builder.add_edge("doc_summarize", "doc_finish")
    builder.add_edge("doc_finish", END)

    return builder.compile()
