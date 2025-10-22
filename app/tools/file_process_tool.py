import io
import cv2
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Literal
from ultralytics import YOLO
from app.logger import logger
from datetime import datetime
from pydantic import PrivateAttr
from app.cores.config import config
from app.tools.base import BaseTool, ToolFailure, ToolResult

_FILE_PROCESS_TOOL = """
执行文档分析和填充任务，根据输入的文档模板内容，自动识别文档中的标题信息，并结合模型生成的内容，对文档进行智能填充。
输入可以是文档路径或文本内容，输出包含填充后的文档及相关元数据。
"""


class FileProcessTool(BaseTool):
    name: str = "file_process_tool"
    description: str = _FILE_PROCESS_TOOL
    parameters: dict = {
        "type": "object",
        "properties": {
            "doc_template_path": {
                "type": "string",
                "description": "待填充的文档模板路径，必须是有效的 DOCX 文件路径。",
            },
            "image_content": {
                "type": "object",
                "description": (
                    "模型分析得到的内容，键为文档标题（可能带编号），"
                    "值为要填充的内容（文本、列表或字典形式）。"
                ),
            },
            "question": {
                "type": "string",
                "description": "本次任务的描述或用户问题，用于指导模型生成内容。",
            },
            "output_path": {
                "type": "string",
                "description": "生成文档的输出路径，必须以 .docx 结尾。",
            },
        },
        "required": ["doc_template_path", "image_content", "question", "output_path"],
        "additionalProperties": False,
    }
    model_path: str = config.model_path / config.YOLO_SEGMENTATION_MODEL_NAME
    conf: float = config.YOLO_SEGMENTATION_CONF
    iou: float = config.YOLO_SEGMENTATION_IOU
    output_path: str = (
        config.file_post_process_path / config.YOLO_SEGMENTATION_OUTPUT_DIR
    )
    # def __init__(self):
    #     # 使用支持分割的YOLO模型
    #     super().__init__()
    #     model_path = config.model_path / config.YOLO_SEGMENTATION_MODEL_NAME
    #     self._model  = YOLO(model_path)
    #     self.conf = config.YOLO_SEGMENTATION_CONF
    #     self.iou = config.YOLO_SEGMENTATION_IOU
    #     self.output_path = config.file_post_process_path / config.YOLO_SEGMENTATION_OUTPUT_DIR
    #     if not self.output_path.exists():
    #         self.output_path.mkdir(parents=True, exist_ok=True)
    #         logger.info(f"分割图像输出目录不存在，已自动创建: {self.output_path}")

    def execute(
        self, image_data: bytes, image_path: str, image_format: str = "jpg"
    ) -> tuple[bytes, list, str]:
        """
        执行图像分割任务。
        Returns:
            tuple[bytes, list, str]: 分割后的图像bytes、分割信息列表和保存路径。
        """
        # === Step 1: Base64 转 numpy 图像 ===
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        model = YOLO(self.model_path)
        # === Step 2: YOLO 分割推理 ===
        results = model.predict(source=image, conf=self.conf, iou=self.iou, save=False)

        result = results[0]
        segmented_image = result.plot()  # 带mask、框、label的图像
        segmentation_info = []

        # === Step 3: 提取检测结果 ===
        for box in result.boxes:
            cls_id = int(box.cls.cpu().numpy())
            conf = float(box.conf.cpu().numpy())
            label = result.names[cls_id]
            xyxy = box.xyxy.cpu().numpy().tolist()[0]

            segmentation_info.append(
                {
                    "class_id": cls_id,
                    "class_name": label,
                    "confidence": conf,
                    "bbox": xyxy,
                }
            )

        # === Step 4: 动态生成输出路径 ===
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"分割图像输出目录不存在，已自动创建: {self.output_path}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_stem = Path(image_path).stem if image_path else "input"
        image_stem = f"{image_stem}_{timestamp}"
        save_path = self.output_path / f"{image_stem}_seg.{image_format}"

        cv2.imwrite(str(save_path), segmented_image)
        logger.info(f"分割结果已保存: {save_path}")

        # === Step 5: 转 bytes 返回 ===
        segmented_bytes = self._image_to_bytes(segmented_image, image_format)
        return segmented_bytes, segmentation_info, str(save_path)

    @staticmethod
    def _image_to_bytes(image: np.ndarray, format: str = "jpg") -> bytes:
        """将numpy图像转换为bytes"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        save_format = "jpeg" if format.lower() in ["jpg", "jpeg"] else format
        pil_image.save(buffer, format=save_format)
        buffer.seek(0)
        return buffer.getvalue()
