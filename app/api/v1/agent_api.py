"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
import os
import traceback
import shutil
from app.logger import logger
from fastapi import Body, Form, UploadFile, File
from typing import List, Optional, Tuple, Dict
from app.cores.config import config
from app.database.utils import KnowledgeFile
from app.tools.utils import thread_pool_executor
from app.agents.supervisor_agent import SupervisorAgent
from fastapi.responses import StreamingResponse

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse
from app.database.kb.milvus_kb_service import MilvusKBService
from app.cores.config import config

router = APIRouter()
agent = SupervisorAgent()
# 定义一个用于存放上传文件的目录
UPLOAD_DIRECTORY = config.temp_file_root


def _parse_files_in_thread(
    files: List[Dict],
    dir: str,
    zh_title_enhance: bool,
    chunk_size: int,
    chunk_overlap: int,
):
    """
    通过多线程将上传的文件保存到对应目录内。
    生成器返回保存结果：[success or error, filename, msg, docs]
    """
    def parse_file(file_data: dict):
        try:
            filename = file_data["filename"]
            file_path = os.path.join(dir, filename)
            file_content = file_data["content"]

            # 写入文件
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file_content)

            # 解析
            kb_file = KnowledgeFile(filename=filename, knowledge_base_name="temp")
            kb_file.filepath = file_path
            docs = kb_file.file2text(
                zh_title_enhance=zh_title_enhance,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return True, filename, f"成功上传文件 {filename}", docs

        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {traceback.format_exc()}"
            logger.error(msg)
            return False, filename, msg, []

    # ✅ 正确传入 file_data_list，而不是 UploadFile 对象
    params = [{"file_data": file_data} for file_data in files]

    # ✅ 使用你自己的线程池封装执行
    for result in thread_pool_executor(parse_file, params=params):
        yield result


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
            file_list = [
                os.path.join(UPLOAD_DIRECTORY, file["saved_path"])
                for file in metadata_dict["files"]
            ]

        # async def event_stream():
        #     async for chunk in agent.chat_response(message=query, file_list=file_list):
        #         yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # return StreamingResponse(event_stream(), media_type="text/event-stream")
        result = await agent.chat_response(message=query, file_list=file_list)
        logger.info(result)
        return result

    except Exception as e:
        import traceback as tra
        logger.error(str(tra.format_exc()))
        raise HTTPException(status_code=500, detail=str(e))


def get_temp_dir(id: str = None) -> Tuple[str, str]:
    """
    创建一个临时目录，返回（路径，文件夹名称）
    """
    import uuid

    if id is not None:  # 如果指定的临时目录已存在，直接返回
        path = os.path.join(config.doc_pre_path, id)
        if os.path.isdir(path):
            return path, id

    id = uuid.uuid4().hex
    path = os.path.join(config.doc_pre_path, id)
    os.mkdir(path)
    return path, id


DOCUMENT_EXTS = {".pdf", ".docx", ".txt", ".md"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif"}


@router.post("/upload", summary="上传文件")
async def upload_files(
    files: List[UploadFile] = File(..., description="一个或多个待上传的文件"),
    prev_id: str = Form(None, description="前知识库ID"),
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    zh_title_enhance: bool = False,
):
    """
    接收一个或多个文件，并将它们保存到服务器的 'temp' 文件夹中。
    如果 'temp' 文件夹不存在，会自动创建。
    """
    # 确保上传目录存在
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)

    path, id = get_temp_dir(prev_id)
    saved_filenames = []
    uploaded_images = []
    failed_files = []
    documents_to_add = []
    file_data_list = []
    milvus_service = MilvusKBService()

    for file in files:
        logger.info(f"正在处理文件 {file.filename},{file}")

        file_ext = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        content = await file.read()  # 用异步读取
        file_data_list.append(
            {
                "filename": file.filename,
                "content": content,
            }
        )
        
        with open(file_path, "wb") as f:
            f.write(content)

        saved_filenames.append(file.filename)
        # try:
        #     with open(file_path, "wb") as buffer:
        #         shutil.copyfileobj(file.file, buffer)
        #     saved_filenames.append(file.filename)
        # except Exception as e:
        #     # 如果任何一个文件保存失败，则返回错误
        #     raise HTTPException(
        #         status_code=500, detail=f"文件 '{file.filename}' 保存失败: {e}"
        #     )
        # finally:
        #     # 确保关闭文件句柄
        #     file.file.close()

        if file_ext in IMAGE_EXTS:
            uploaded_images.append(file.filename)

        if file_ext in DOCUMENT_EXTS:
           
            # 文档进行向量化
            for success, file, msg, docs in _parse_files_in_thread(
                files=file_data_list,
                dir=path,
                zh_title_enhance=zh_title_enhance,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ):
                if success:
                    documents_to_add += docs
                else:
                    failed_files.append({file: msg})
        else:
            # 其他文件类型暂时只保存，不向量化
            uploaded_images.append(file.filename)

        if documents_to_add:

            try:
                doc_infos = milvus_service.do_add_doc(documents_to_add)
                logger.info(f"存储的文件为: {doc_infos}")
            except Exception as e:
                import traceback as tra
                logger.error(f"无法链接到Milvus服务器: {tra.format_exc()}")

    return {
        "message": f"成功上传 {len(saved_filenames)} 个文件。",
        "uploaded_files": saved_filenames,
    }


