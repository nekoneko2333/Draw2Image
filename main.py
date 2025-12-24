"""
实时草图转图像系统 - FastAPI 后端 (用户意图增强版)

移除自动语义识别 (BLIP)，完全尊重用户输入的 Prompt。
保留双 ControlNet 控制和照片模式深度图反馈。
"""

import os
import base64
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn

# ============== 环境配置 ==============
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from processor import create_processor, ClassicNSTProcessor

app = FastAPI(title="人机交互艺术创作系统", version="1.2.0")

# 初始化处理器（此时内部 BLIP 已删，但保留了翻译和 IP-Adapter 逻辑）
processor = create_processor("ai")

# ============== WebSocket 实时通道 ==============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ 实时创作通道连接成功")
    
    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
                
                # 1. 提取用户输入的参数（保留 Prompt）
                image_base64 = data.get('image', '')
                style = data.get('style', 'vangogh')
                prompt = data.get('prompt', '')  # 用户的核心意图
                strength = float(data.get('strength', 0.6))
                ref_image_name = data.get('ref_image_name', None)
                is_photo_mode = data.get('is_photo_mode', False) # 保留照片模式开关
                
                # 2. 图像解码
                img = processor.decode_base64_image(image_base64)
                
                if img is not None and img.size > 0:
                    # 3. 推理：透传所有用户参数
                    # 此时 processor.process 内部会处理用户 Prompt 的翻译
                    processed_img = processor.process(
                        image=img,
                        style=style,
                        prompt=prompt,
                        strength=strength,
                        ref_image_name=ref_image_name,
                        is_photo_mode=is_photo_mode
                    )
                    
                    # 4. 获取结果图
                    result_base64 = processor.encode_image_to_base64(processed_img)
                    
                    # 5. 【核心保留】如果处于照片模式，返回深度图预览
                    depth_base64 = None
                    if is_photo_mode and hasattr(processor, 'get_last_depth_image_base64'):
                        depth_base64 = processor.get_last_depth_image_base64()
                    
                    # 打包返回
                    if depth_base64:
                        await websocket.send_text(json.dumps({
                            "type": "result",
                            "image": result_base64,
                            "depth": depth_base64
                        }))
                    else:
                        await websocket.send_text(result_base64)
                        
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        logger.warning("🔌 客户端断开")
    except Exception as e:
        logger.error(f"❌ 系统异常: {e}")

# ============== 其他 API (保持不变) ==============

@app.get("/api/assets")
async def get_assets_structure():
    return {"assets": processor.get_assets_structure(), "base_url": "/assets"}

# 挂载静态资源
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="/root/autodl-tmp/assets/styles"), name="assets")

# 精修渲染接口 (NST)
nst_processor = None
@app.post("/api/render_refined")
async def render_refined(request: dict):
    global nst_processor
    if nst_processor is None:
        nst_processor = ClassicNSTProcessor()
    
    try:
        img = processor.decode_base64_image(request.get('image', ''))
        
        # 提取高级参数
        preserve_color = request.get('preserve_color', False)
        pool_type = request.get('pool_type', 'max')
        style_weight = float(request.get('style_weight', 0.05))
        
        result_img = nst_processor.process(
            image=img,
            style=request.get('style', 'vangogh'),
            ref_image_name=request.get('ref_image_name', None),
            preserve_color=preserve_color,
            pool_type=pool_type,
            style_weight=style_weight
        )
        return {"image": processor.encode_image_to_base64(result_img, quality=90)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)