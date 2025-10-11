from fastapi import APIRouter
from app.logger import logger
from app.api.v1.agent_api import router as agent_router

api_router = APIRouter()
api_router.include_router(agent_router, prefix="/chat", tags=["chat"])