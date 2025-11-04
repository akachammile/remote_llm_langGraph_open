"""This file contains the main application entry point."""

import os
import uvicorn
import argparse

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.database.db.model.message_model import MessageModel

from app.api.v1.api import api_router
from app.logger import logger
from app.database.base import engine, Base
logger.info(f"main.py 中 Base 对象的 ID: {id(Base)}")
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理生命周期"""
    logger.info(
        "程序启动",
        project_name="1",
        version="settings.VERSION",
        api_prefix="settings.API_V1_STR",
    )
    yield
    logger.info("程序中止")

def create_app():
    app = FastAPI(
        title="一体机后端",
        version="v1",
        description="一体机后端自动化",
        lifespan=lifespan,
    )

    # 设置跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源，生产环境应限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")

    return app

app = create_app()


def init_db():
    logger.info("正在初始化数据库...")
    try:
        # --- 诊断点 1: 检查 Base 是否发现模型 ---
        # 在 Base.metadata.create_all(engine) 之前添加
        logger.info(f"当前 engine 连接的数据库 URL: {engine.url}")
        logger.info(f"init_db 中 Base 对象的 ID: {id(Base)}") # <--- 添加这行
        logger.info(f"Base 发现的模型数量: {len(Base.metadata.tables)}")
        logger.info(f"Base 发现的模型名称: {list(Base.metadata.tables.keys())}")

          # 这行代码会根据所有继承自 Base 的模型定义，在数据库中创建对应的表
        Base.metadata.create_all(engine)
        logger.info("数据库初始化完成！！！")
    except Exception as e:
        # --- 诊断点 2: 打印详细错误信息 ---
        logger.error(f"数据库初始化失败: {e}", exc_info=True) # exc_info=True 会打印完整的堆栈信息

def run_api(host, port, **kwargs):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="langchain", description="基于本地知识库的多模态问答")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)
    init_db()
    run_api(
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )