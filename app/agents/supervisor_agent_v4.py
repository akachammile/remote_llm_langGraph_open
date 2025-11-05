from typing import TypedDict, Optional, List, Union, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app.agents.base import BaseAgent
from app.agents.vision_node import build_vision_subgraph
from app.agents.doc_node import build_doc_subgraph
from app.agents.chat_node import build_chat_subgraph
from app.schemas.schema import Message
from app.logger import logger


class SupervisorState(TypedDict, total=False):
    """Supervisor母图状态，字段命名与子图完全不同"""
    user_question: Optional[str]
    user_image_b64: Optional[str]
    conversation_messages: List[BaseMessage]
    final_image_paths: List[str]
    final_document_paths: List[str]
    routing_target: Optional[str]
    supervisor_meta: Dict[str, Any]
    next_agent: Optional[str]
    executed_agents: List[str]  # 记录已执行的Agent，避免重复执行


class SupervisorAgent(BaseAgent):
    """V4版Supervisor，字段分离架构：在节点内调用子图并做字段映射"""

    name: str = "SupervisorAgentV4"

    def __init__(self):
        super().__init__(
            name="SupervisorAgentV4",
            description="母图(字段分离版),在节点中调用子图并做字段映射",
            system_prompt="",
            next_step_prompt=""
        )
        self._graph: Optional[CompiledStateGraph] = None
        self.vision_subgraph: CompiledStateGraph = build_vision_subgraph()
        self.doc_subgraph: CompiledStateGraph = build_doc_subgraph()
        self.chat_subgraph: CompiledStateGraph = build_chat_subgraph()
        logger.info("SupervisorAgent初始化完成，Vision/Doc/Chat子图已预加载")

    async def create_supervisor_graph(self) -> CompiledStateGraph:
        """Supervisor母图构建"""
        if self._graph is not None:
            return self._graph

        builder = StateGraph(SupervisorState)

        async def top_supervisor(state: SupervisorState) -> Dict[str, Any]:
            """决策层：使用LLM推理路由到哪个子图"""
            msgs = state.get("conversation_messages", [])
            user_q = state.get("user_question") or ""
            routing = state.get("routing_target")
            executed_agents = state.get("executed_agents", [])
            
            # 关键改进：判断是否应该结束
            # 1. routing为None（子图清空了）
            # 2. 有子图结果（至少执行过一个子图）
            # 3. LLM决策不再需要更多子图（通过提问判断）
            if routing is None and len(executed_agents) > 0:
                # 已经执行过子图，问下LLM是否需要继续
                try:
                    system_msg = Message.system_message(
                        """你是一个任务完成检测系统。根据用户问题和已执行的Agent，判断是否需要继续调用其他Agent。
                                    可用的Agent：
                                    - VisionAgent: 处理图像相关任务
                                    - DocAgent: 处理文档相关任务
                                    - ChatAgent: 处理普通对话
                                    - 

                                    输出要求，下一个任务的Agent以list的形式输出
                                    输出格式要求：[VisionAgent, DocAgent, DocAgent, END]
                                    如果还需要其他Agent，请直接返回Agent名称（VisionAgent/DocAgent/ChatAgent）。"""
                                                        )
                    
                    check_msg = Message.user_message(
                        f"用户问题：{user_q}\n已执行Agent：{', '.join(executed_agents)}\n是否需要继续？"
                    )
                    
                    llm_response = await self.llm.ask_tool_v3(
                        messages=[check_msg],
                        system_msgs=[system_msg],
                        stream=False
                    )
                    
                    # 解析响应
                    if isinstance(llm_response, str):
                        response_text = llm_response
                    elif llm_response and hasattr(llm_response, 'content'):
                        response_text = str(llm_response.content)  # type: ignore
                    else:
                        response_text = str(llm_response) if llm_response else "DONE"
                    
                    if "DONE" in response_text or "完成" in response_text:
                        logger.info(f"LLM判断任务已完成，进入聚合层")
                        return {
                            "conversation_messages": msgs,
                            "routing_target": None,
                            "executed_agents": executed_agents,
                            "supervisor_meta": {"node": "top_supervisor", "status": "task_complete"},
                        }
                    elif "VisionAgent" in response_text and "VisionAgent" not in executed_agents:
                        routing = "VisionAgent"
                        logger.info(f"LLM决策继续调用: {routing}")
                    elif "DocAgent" in response_text and "DocAgent" not in executed_agents:
                        routing = "DocAgent"
                        logger.info(f"LLM决策继续调用: {routing}")
                    elif "ChatAgent" in response_text and "ChatAgent" not in executed_agents:
                        routing = "ChatAgent"
                        logger.info(f"LLM决策继续调用: {routing}")
                    else:
                        # 默认结束
                        logger.info("无法解析继续意图，默认结束")
                        return {
                            "conversation_messages": msgs,
                            "routing_target": None,
                            "executed_agents": executed_agents,
                            "supervisor_meta": {"node": "top_supervisor", "status": "default_complete"},
                        }
                except Exception as e:
                    logger.error(f"任务完成检测失败: {e}，默认结束")
                    return {
                        "conversation_messages": msgs,
                        "routing_target": None,
                        "executed_agents": executed_agents,
                        "supervisor_meta": {"node": "top_supervisor", "status": "error_complete"},
                    }
            
            # 第一次进入，添加用户消息
            if not any(isinstance(msg, HumanMessage) and msg.content == user_q for msg in msgs):
                user_msg = HumanMessage(content=user_q)
                msgs = msgs + [user_msg]

            if not routing:
                try:
                    # 使用BaseAgent附带的llm属性进行LLM推理
                    system_msg = Message.system_message(
                        """你是一个智能路由决策系统，根据用户问题选择最合适的Agent。
                        可用的Agent：
                        - VisionAgent: 处理图像相关任务（分割、识别、分析）
                        - DocAgent: 处理文档相关任务（检索、生成、总结）
                        - ChatAgent: 处理普通对话交互

                        请直接返回Agent名称（VisionAgent/DocAgent/ChatAgent），不要额外解释。"""
                                            )
                                            
                    user_msg_content = f"用户问题：{user_q}"
                    if state.get("user_image_b64"):
                        user_msg_content += "（包含图像数据）"
                    
                    user_msg = Message.user_message(user_msg_content)
                    
                    llm_response = await self.llm.ask_tool_v3(
                        messages=[user_msg],
                        system_msgs=[system_msg],
                        stream=False
                    )
                    
                    # 确保routing_text是字符串
                    if isinstance(llm_response, str):
                        routing_text = llm_response
                    elif llm_response and hasattr(llm_response, 'content'):
                        routing_text = str(llm_response.content)  # type: ignore
                    else:
                        routing_text = str(llm_response) if llm_response else "ChatAgent"
                    
                    # 解析LLM响应
                    if "VisionAgent" in routing_text:
                        routing = "VisionAgent"
                    elif "DocAgent" in routing_text:
                        routing = "DocAgent"
                    elif "ChatAgent" in routing_text:
                        routing = "ChatAgent"
                    else:
                        # 默认路由到Chat
                        routing = "ChatAgent"
                    
                    logger.info(f"LLM路由决策: {routing_text} -> {routing}")
                except Exception as e:
                    logger.error(f"LLM路由决策失败: {e}，使用默认逻辑")
                    # 降级到关键词匹配
                    if state.get("user_image_b64") or "图" in user_q or "分割" in user_q:
                        routing = "VisionAgent"
                    elif "文档" in user_q or "检索" in user_q or "生成" in user_q:
                        routing = "DocAgent"
                    else:
                        routing = "ChatAgent"

            logger.info(f"Supervisor最终决策: 路由={routing}, 问题={user_q[:30]}, 已执行={executed_agents}")

            return {
                "conversation_messages": msgs,
                "routing_target": routing,
                "executed_agents": executed_agents,
                "supervisor_meta": {"node": "top_supervisor", "status": "routed"},
            }

        async def invoke_vision_subgraph(state: SupervisorState) -> Dict[str, Any]:
            """调用Vision子图，字段映射：母图 -> 子图"""
            logger.info("开始调用 Vision 子图，执行字段映射...")

            # 母图 -> 子图的字段映射
            vision_input = {
                "input_query": state.get("user_question"),
                "input_image_base64": state.get("user_image_b64"),
                "input_image_file": None,
                "internal_messages": [],
                "segmentation_result": None,
                "output_image_files": [],
                "output_description": None,
                "execution_meta": {"caller": "supervisor_v4"},
            }

            # 调用子图
            sub_result = await self.vision_subgraph.ainvoke(vision_input)
            logger.info(f"Vision 子图执行完成，输出文件数: {len(sub_result.get('output_image_files', []))}")

            # 子图 -> 母图的字段映射与合并
            mother_msgs = state.get("conversation_messages", [])
            sub_msgs = sub_result.get("internal_messages", [])
            merged_msgs = mother_msgs + sub_msgs

            mother_imgs = state.get("final_image_paths", [])
            sub_imgs = sub_result.get("output_image_files", [])
            merged_imgs = mother_imgs + sub_imgs

            sub_desc = sub_result.get("output_description") or "子图已完成"
            desc_msg = AIMessage(content=f"[Vision子图结果] {sub_desc}")
            merged_msgs = merged_msgs + [desc_msg]

            logger.info(f"字段合并完成：消息{len(merged_msgs)}条，图像{len(merged_imgs)}个")

            # 记录已执行的Agent
            executed = state.get("executed_agents", [])
            if "VisionAgent" not in executed:
                executed = executed + ["VisionAgent"]

            return {
                "conversation_messages": merged_msgs,
                "final_image_paths": merged_imgs,
                "routing_target": None,
                "executed_agents": executed,
                "supervisor_meta": {"node": "invoke_vision_subgraph", "status": "merged"},
            }

        async def aggregator(state: SupervisorState) -> Dict[str, Any]:
            """聚合层：产生最终结果"""
            final_reply = "任务已完成"
            imgs = state.get("final_image_paths", [])
            docs = state.get("final_document_paths", [])

            if imgs:
                final_reply += f"\n\u5904理了 {len(imgs)} 个图像文件。"
            if docs:
                final_reply += f"\n生成了 {len(docs)} 个文档文件。"

            import json
            final_ai_msg = AIMessage(
                content=json.dumps({
                    "reply": final_reply,
                    "processed_image_path": imgs,
                    "processed_doc_path": docs,
                })
            )

            msgs = state.get("conversation_messages", []) + [final_ai_msg]
            logger.info("聚合层已生成最终回复")

            return {
                "conversation_messages": msgs,
                "supervisor_meta": {"node": "aggregator", "status": "complete"},
            }

        async def invoke_doc_subgraph(state: SupervisorState) -> Dict[str, Any]:
            """调用Doc子图，字段映射：母图 -> 子图"""
            logger.info("开始调用 Doc 子图，执行字段映射...")

            # 母图 -> 子图的字段映射
            doc_input = {
                "input_user_query": state.get("user_question"),
                "input_doc_files": [],
                "input_context": None,
                "doc_messages": [],
                "retrieval_results": None,
                "generated_content": None,
                "output_doc_files": [],
                "output_summary": None,
                "doc_meta": {"caller": "supervisor_v4"},
            }

            # 调用子图
            sub_result = await self.doc_subgraph.ainvoke(doc_input)
            logger.info(f"Doc 子图执行完成，输出文件数: {len(sub_result.get('output_doc_files', []))}")

            # 子图 -> 母图的字段映射与合并
            mother_msgs = state.get("conversation_messages", [])
            sub_msgs = sub_result.get("doc_messages", [])
            merged_msgs = mother_msgs + sub_msgs

            mother_docs = state.get("final_document_paths", [])
            sub_docs = sub_result.get("output_doc_files", [])
            merged_docs = mother_docs + sub_docs

            sub_summary = sub_result.get("output_summary") or "子图已完成"
            desc_msg = AIMessage(content=f"[Doc子图结果] {sub_summary}")
            merged_msgs = merged_msgs + [desc_msg]

            logger.info(f"字段合并完成：消息{len(merged_msgs)}条，文档{len(merged_docs)}个")

            # 记录已执行的Agent
            executed = state.get("executed_agents", [])
            if "DocAgent" not in executed:
                executed = executed + ["DocAgent"]

            return {
                "conversation_messages": merged_msgs,
                "final_document_paths": merged_docs,
                "routing_target": None,
                "executed_agents": executed,
                "supervisor_meta": {"node": "invoke_doc_subgraph", "status": "merged"},
            }

        async def invoke_chat_subgraph(state: SupervisorState) -> Dict[str, Any]:
            """调用Chat子图，字段映射：母图 -> 子图"""
            logger.info("开始调用 Chat 子图，执行普通对话...")

            # 母图 -> 子图的字段映射
            chat_input = {
                "user_input": state.get("user_question"),
                "chat_history": [],
                "chat_response": None,
                "chat_meta": {"caller": "supervisor_v4"},
            }

            # 调用子图
            sub_result = await self.chat_subgraph.ainvoke(chat_input)
            logger.info(f"Chat 子图执行完成")

            # 子图 -> 母图的字段映射与合并
            mother_msgs = state.get("conversation_messages", [])
            sub_msgs = sub_result.get("chat_history", [])
            merged_msgs = mother_msgs + sub_msgs

            chat_response = sub_result.get("chat_response") or "子图已完成"
            desc_msg = AIMessage(content=f"[Chat子图结果] {chat_response}")
            merged_msgs = merged_msgs + [desc_msg]

            logger.info(f"字段合并完成：消息{len(merged_msgs)}条")

            # 记录已执行的Agent
            executed = state.get("executed_agents", [])
            if "ChatAgent" not in executed:
                executed = executed + ["ChatAgent"]

            return {
                "conversation_messages": merged_msgs,
                "routing_target": None,
                "executed_agents": executed,
                "supervisor_meta": {"node": "invoke_chat_subgraph", "status": "merged"},
            }

        def route_decision(state: SupervisorState) -> str:
            """路由决策"""
            target = state.get("routing_target")
            if target == "VisionAgent":
                return "invoke_vision_subgraph"
            elif target == "DocAgent":
                return "invoke_doc_subgraph"
            elif target == "ChatAgent":
                return "invoke_chat_subgraph"
            return "aggregator"

        builder.add_node("top_supervisor", top_supervisor)
        builder.add_node("invoke_vision_subgraph", invoke_vision_subgraph)
        builder.add_node("invoke_doc_subgraph", invoke_doc_subgraph)
        builder.add_node("invoke_chat_subgraph", invoke_chat_subgraph)
        builder.add_node("aggregator", aggregator)

        builder.add_edge(START, "top_supervisor")
        builder.add_conditional_edges(
            "top_supervisor",
            route_decision,
            ["invoke_vision_subgraph", "invoke_doc_subgraph", "invoke_chat_subgraph", "aggregator"],
        )
        builder.add_edge("invoke_vision_subgraph", "top_supervisor")
        builder.add_edge("invoke_doc_subgraph", "top_supervisor")
        builder.add_edge("invoke_chat_subgraph", "top_supervisor")
        builder.add_edge("aggregator", END)

        self._graph = builder.compile()
        logger.info("Supervisor 母图构建完成")
        return self._graph

    # async def chat_response(
    #     self, question: str, image_b64: Optional[str] = None
    # ) -> SupervisorState:
    #     """外部调用入口"""
    #     graph = await self.create_supervisor_graph()
    #     init_state: SupervisorState = {
    #         "user_question": question,
    #         "user_image_b64": image_b64,
    #         "conversation_messages": [],
    #         "final_image_paths": [],
    #         "final_document_paths": [],
    #         "routing_target": None,
    #         "supervisor_meta": {},
    #     }
    #     logger.info(f"开始执行任务: {question[:50]}")
    #     result = await graph.ainvoke(init_state)
    #     logger.info("任务执行完成")
    #     return result  # type: ignore
