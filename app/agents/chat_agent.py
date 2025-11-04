import json
from typing import Dict,List
from app.logger import logger
from collections import defaultdict
from app.agents.base import BaseAgent
from app.schemas.schema import Message
from app.graphs.graph_state import AgentState
from langgraph.graph import StateGraph, START, END
from app.prompts.chat_prompt import SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from app.tools.image_segmentation_tool import ImageSegmentationTool
from app.database.db.repository.message_repository import (
    add_message_to_db,
    update_message,
    get_message_by_id,
)
from fastapi.responses import StreamingResponse

class ChatAgent(BaseAgent):
    name: str = "ChatAgent"
    description: str = "多模态智能问答模块，支持对话以及对文档、图像等多模态文件进行内容上的分析解读"
    tool: Dict[str, str] = defaultdict(str)
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = ""

    async def chat_supervisor_agent(self, state: AgentState) -> AgentState:
        """
        Description: 进行对话工作

        Args:
            state: 包含消息的当前代理状态

        Returns:
            AgentState.
        """
        try:
            logger.info(f"对话系统记忆:{self.memory.messages}")
            question: str = state["question"]
            memory: str = state["memory"]

            history: List[Message] = self.memory.get_recent_messages(n=30)
            prompt = self.system_prompt.format(question=question, history=history + state["memory"])
            user_message = Message.user_message(question) 
            system_msgs = Message.system_message(prompt)

            chat_reuslt = await self.llm.ask_v2([user_message], [system_msgs])
            logger.info(f"对话结果: {chat_reuslt}")
            # plan_results = await self.llm.ask_v2([user_message], system_msgs=[system_msgs])

            # state["reflection"] = True
            if state["messages"]:
                self.memory.add_message(
                    Message.assistant_message(content=state["messages"].content)
                )
            return state
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
    
  
    
    def build_subgraph(self):
        # 创建子图
        chat_subgraph = StateGraph(AgentState)
        chat_subgraph.add_node("chat", self.chat_supervisor_agent)
        chat_subgraph.add_edge(START, "chat")
        chat_subgraph.add_edge("chat", END) 
        chat_subgraph = chat_subgraph.compile()
        return chat_subgraph

    # async def chat(self, state: AgentState) -> StreamingResponse:
    #     """
    #     Description: 进行对话工作并返回流式响应

    #     Args:
    #         state: 包含消息的当前代理状态

    #     Returns:
    #         StreamingResponse.
    #     """
    #     async def event_generator():
    #         """Generate streaming events.

    #         Yields:
    #             str: Server-sent events in JSON format.

    #         Raises:
    #             Exception: If there's an error during streaming.
    #         """
    #         try:
    #             full_response = ""
    #             question: str = state["question"]
    #             history = self.memory.get_recent_messages(n=10)
    #             prompt = self.system_prompt.format(
    #                 question=question, history=history + state["memory"]
    #             )
    #             async for chunk in await self.llm.ask_stream(prompt, state):
    #                 full_response += chunk
    #                 response = StreamResponse(content=chunk, done=False)
    #                 yield f"data: {json.dumps(response.model_dump())}\n\n"

    #             # Save the complete response to memory and database
    #             if full_response:
    #                 self.memory.add_message(
    #                     Message.assistant_message(content=full_response)
    #                 )
    #                 # TODO: Save message to database
    #                 # add_message_to_db()

    #             # Send final message indicating completion
    #             final_response = StreamResponse(content="", done=True)
    #             yield f"data: {json.dumps(final_response.model_dump())}\n\n"

    #         except Exception as e:
    #             logger.error("stream_chat_request_failed")
    #             error_response = StreamResponse(content=str(e), done=True)
    #             yield f"data: {json.dumps(error_response.model_dump())}\n\n"

    #     return StreamingResponse(event_generator(), media_type="text/event-stream")
    

    # def build_subgraph(self):
    #     chat_subgraph = StateGraph(AgentState)
    #     chat_subgraph.add_node("chat", self.chat)
    #     chat_subgraph.add_edge(START, "chat")
    #     chat_subgraph.add_edge("chat", END)
    #     chat_subgraph = chat_subgraph.compile()
    #     return chat_subgraph