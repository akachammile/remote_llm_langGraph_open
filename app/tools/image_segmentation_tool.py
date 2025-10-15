import io
import cv2
import base64
import numpy as np
from PIL import Image
from typing import Literal
from ultralytics import YOLO
from app.logger import logger
from app.cores.config import config
from langchain_core.tools import tool


class ImageSegmentationTool:

    def __init__(self):
        # 使用支持分割的YOLO模型
        model_path = config.model_path / "yolov8n-seg.pt"
        self.model = YOLO(model_path)

    def _image_to_bytes(self, image: np.ndarray, format: str = 'jpg') -> bytes:
        """将numpy图像转换为bytes"""
        # OpenCV使用BGR，转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image_rgb)
        
        # 保存到内存
        buffer = io.BytesIO()
        
        # Pillow uses 'JPEG' for .jpg files
        save_format = 'jpeg' if format.lower() in ['jpg', 'jpeg'] else format
            
        pil_image.save(buffer, format=save_format)
        buffer.seek(0)
        
        return buffer.getvalue()
        
    def segment_image(self, image_data: bytes, image_path:str, image_format: str = "jpg") -> tuple[bytes, list, str]:
        """执行图像分割"""
        # 将bytes转换为numpy数组
        img_bytes = base64.b64decode(image_data)  # 得到原始图片二进制bytes

        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)        
        # YOLO分割推理
        results = self.model(source=image, conf=0.2)
        
        # 创建分割后的图像
        segmented_image = image.copy()
        segmentation_info = []
        # TODO 这里需要修改，路径不可写死
        path = r"E:\\1_LLM_PROJECT\\remote_llm_langGraph\\"+ image_path+"__result." + image_format
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for i, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confidences)):
                    class_name = result.names[int(cls)]
                    
                    
                    # 应用分割掩码
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    mask_3d = np.stack([mask_resized] * 3, axis=2)
                    
                    # 为每个分割对象生成随机颜色
                    color = np.random.randint(0, 255, 3)
                    colored_mask = mask_3d * color
                    
                    # 将分割结果叠加到原图
                    segmented_image = cv2.addWeighted(segmented_image, 1, colored_mask.astype(np.uint8), 0.5, 0)
                    
                    # 绘制边界框和标签
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(segmented_image, (x1, y1), (x2, y2), color.tolist(), 2)
                    
                    # 添加标签
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(segmented_image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
                    cv2.imwrite(path, segmented_image)

                    segmentation_info.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'mask': mask_resized.tolist()
                    })
        cv2.imwrite(path, segmented_image)
        # 将分割后的图像转换为bytes
        segmented_bytes = self._image_to_bytes(segmented_image, image_format)
        
        return segmented_bytes, segmentation_info, path
    
    