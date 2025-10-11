"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""
import json
import os
import shutil
from app.logger import logger
from fastapi import Body, Form, UploadFile, File
from langgraph.types import Command
from app.agents.base import BaseAgent
from langchain_ollama import ChatOllama
from typing import List, Literal, Optional

from IPython.display import Image, display

from app.graphs.graph_state import AgentState
from app.agents.vision_agent import VisionAgent
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from app.agents.supervisor_agent import SupervisorAgent
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState, START, END
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse


router = APIRouter()
agent = SupervisorAgent()
# 定义一个用于存放上传文件的目录
UPLOAD_DIRECTORY = "temp"

@router.post("/chat")
async def chat(
    query: str = Body(..., description="用户问题"),
    metadata: str = Body(..., description="用户附件，JSON 字符串"),
    stream: Optional[bool] = Body(True, description="流式输出，字符串形式"),
    # files: Optional[List[UploadFile]] = File(None)
    # session: Session = Depends(get_current_session),
):
    """chat接口
    Args:
        query (str, optional): 用户提问. Defaults to Body(..., description="用户问题").
        metadata (dict, optional): 元数据. Defaults to Body({}, description="用户附件").
        stream (bool, optional): 流式输出. Defaults to Body(True, description="流式输出").

    Raises:
        HTTPException: _description_
    """

    try:
        metadata_dict = json.loads(metadata) if metadata else {}
        file_list = []
    except Exception as e:
        logger.exception(f"解析文件出错：{e}")
        metadata_dict = {}
    # stream 可能是字符串，强转为 bool
    if isinstance(stream, str):
        stream_bool = stream.lower() in ("1", "true", "yes")
    else:
        stream_bool = bool(stream)
        
    try:
        if metadata_dict:
            file_list = [os.path.join(UPLOAD_DIRECTORY,file["saved_path"]) for file in metadata_dict["files"]]
        
        result = await agent.chat_response(message = query, file_list = file_list)
        logger.info(result)
        return result

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/upload", summary="上传文件")
async def upload_files(
    files: List[UploadFile] = File(..., description="一个或多个待上传的文件")
):
    """
    接收一个或多个文件，并将它们保存到服务器的 'temp' 文件夹中。
    如果 'temp' 文件夹不存在，会自动创建。
    """
    # 确保上传目录存在
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)

    saved_filenames = []
    for file in files:
        # 构建安全的文件路径
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        
        try:
            # 使用 shutil.copyfileobj 来高效地保存文件，适合处理大文件
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_filenames.append(file.filename)
        except Exception as e:
            # 如果任何一个文件保存失败，则返回错误
            raise HTTPException(
                status_code=500, 
                detail=f"文件 '{file.filename}' 保存失败: {e}"
            )
        finally:
            # 确保关闭文件句柄
            file.file.close()

    return {
        "message": f"成功上传 {len(saved_filenames)} 个文件。",
        "uploaded_files": saved_filenames
    }
