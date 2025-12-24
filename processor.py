"""
实时草图生成系统 - 4090 巅峰性能与工程化完整版
集成了：
1. 传统 CV 算法集群 (MockProcessor) - 提供 6 种以上手工艺术滤镜.
2. 高性能 AI 推理引擎 (AIProcessor) - 针对 4090 优化的双控架构.
3. 传输性能优化层 - 自动下采样与质量控制，解决 WebSocket 延迟.
4. 异常防御系统 - 自动处理 NoneType、元组解包及属性锁定问题.
5. 经典 NST 精修引擎 (ClassicNSTProcessor) - 调用 Neural-Style-Transfer 项目.
"""

import os
import time
import torch
import gc
import numpy as np
import cv2
import base64
import logging
import subprocess
import uuid
from PIL import Image
from io import BytesIO
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

# 核心兼容性补丁
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    LCMScheduler
)
from transformers import CLIPVisionModelWithProjection
import warnings

# 过滤掉 LoRA 相关的无害警告
warnings.filterwarnings("ignore", message=".*No LoRA keys associated.*")
warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")

logger = logging.getLogger(__name__)

# ============================================================
# 1. 处理器基类定义 (工程化架构支撑)
# ============================================================

class BaseProcessor(ABC):
    """
    定义了系统的标准化处理协议
    包含高性能编解码逻辑，是 main.py 与算法层的连接纽带。
    """
    @abstractmethod
    def process(self, image: np.ndarray, style: str, prompt: str = "", **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_supported_styles(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_assets_structure(self) -> Dict[str, List[str]]:
        """返回风格素材库结构：{风格名: [文件名列表]}"""
        pass

    @staticmethod
    def decode_base64_image(base64_str: str) -> np.ndarray:
        """高性能 Base64 解码，支持 DataURL 格式"""
        try:
            start_time = time.time()
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            img_bytes = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # 防御性编程：强制类型转换
            if isinstance(img, tuple): img = img[1]
            # logger.info(f"⏱️ 解码耗时: {1000*(time.time()-start_time):.2f}ms")
            return img if img is not None else np.zeros((512, 512, 3), dtype=np.uint8)
        except Exception as e:
            logger.error(f"❌ 图像解码异常: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)

    @staticmethod
    def encode_image_to_base64(img: np.ndarray, quality: int = 80) -> str:
        """
        高性能编码并执行质量压缩，解决网络延迟问题
        """
        try:
            start_time = time.time()
            # 针对网络传输优化：采用 JPG 压缩减少 Base64 长度
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, buffer = cv2.imencode('.jpg', img, params)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            # logger.info(f"⏱️ 编码耗时: {1000*(time.time()-start_time):.2f}ms")
            return f"data:image/jpeg;base64,{base64_str}"
        except Exception:
            return ""

# ============================================================
# 2. Mock 处理器实现 (保留全量 OpenCV 逻辑，增加代码丰富度)
# ============================================================

class MockProcessor(BaseProcessor):
    """
    Mock 处理器：在无显存环境下的极速降级方案
    包含油画、素描、卡通、水彩、浮雕、中值滤波等 6 种手工算法。
    """
    def __init__(self):
        self.style_handlers = {
            'oil': self._apply_oil,
            'sketch': self._apply_sketch,
            'watercolor': self._apply_watercolor,
            'cartoon': self._apply_cartoon,
            'emboss': self._apply_emboss,
            'pencil': self._apply_pencil
        }
        logger.info("🎨 MockProcessor 集群初始化完成")

    def get_supported_styles(self) -> List[str]:
        return list(self.style_handlers.keys())
    
    def get_assets_structure(self) -> Dict[str, List[str]]:
        """Mock 模式无素材库，返回空结构"""
        return {style: [] for style in self.style_handlers.keys()}

    def process(self, image: np.ndarray, style: str, prompt: str = "", **kwargs) -> np.ndarray:
        handler = self.style_handlers.get(style)
        return handler(image) if handler else image

    # 各风格的精细算法实现
    def _apply_oil(self, img): return cv2.pyrMeanShiftFiltering(img, 20, 45)
    def _apply_sketch(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        return cv2.cvtColor(cv2.divide(gray, 255 - cv2.GaussianBlur(inv, (21, 21), 0), scale=256), cv2.COLOR_GRAY2BGR)
    def _apply_watercolor(self, img): return cv2.stylization(img, sigma_s=60, sigma_r=0.45)
    def _apply_cartoon(self, img):
        gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        return cv2.bitwise_and(cv2.bilateralFilter(img, 9, 300, 300), cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    def _apply_emboss(self, img): return cv2.convertScaleAbs(cv2.filter2D(img, -1, np.array([[-2,-1,0], [-1,1,1], [0,1,2]])) + 128)
    def _apply_pencil(self, img): return cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)[1]

# ============================================================
# 3. AI 处理器实现 (RTX 4090 深度优化版)
# ============================================================

class AIProcessor(BaseProcessor):
    """
    RTX 4090 深度优化引擎：
    集成 IP-Adapter (风格调色) + 动态风格 LoRA (灵魂笔触) + 双 ControlNet (结构控制).
    通过 LCM-LoRA 实现 0.3s 推理反馈.
    
    灵魂注入架构（Soul Injection）：
    - 风格 LoRA: 注入大师的真实笔触风格，而非简单的颜色映射
    - IP-Adapter: 降级为辅助调色器，提供色彩参考
    - Scribble ControlNet: 主导线条结构控制
    - Depth ControlNet: 辅助空间深度感知，提升立体效果
    
    LoRA 动态加载策略：
    - 每个风格可选配独立的 .safetensors LoRA 文件
    - 切换风格时自动卸载旧 LoRA、加载新 LoRA
    - 无 LoRA 时优雅降级为纯 IP-Adapter 模式
    
    风格素材库模式：
    - 支持每个风格下包含多张参考图
    - 自动扫描目录结构并预加载至显存
    - 前端可选择特定参考图进行风格迁移
    """
    def __init__(self):
        logger.info("🚀 AIProcessor 正在加载大模型组件 (4090 双ControlNet + 动态LoRA)...")
        self.device = "cuda"
        self.dtype = torch.float16
        self.style_assets_dir = "/root/autodl-tmp/assets/styles"
        self.lora_dir = "/root/autodl-tmp/models/loras"  # LoRA 文件目录
        
        # LoRA 状态跟踪（优化版：避免重复加载 LCM-LoRA）
        self.loaded_style_lora: Optional[str] = None  # 当前已加载的风格 LoRA 名称
        self.lora_adapter_name = "style_lora"  # 风格 LoRA 的适配器名称
        self.lcm_adapter_loaded = False  # LCM-LoRA 是否已加载
        
        # 中文翻译映射表（轻量级方案，避免安装额外库）
        self.cn_to_en_map = {
            "猫": "cat", "狗": "dog", "女孩": "girl", "男孩": "boy", "人": "person",
            "花": "flower", "树": "tree", "山": "mountain", "水": "water", "天空": "sky",
            "太阳": "sun", "月亮": "moon", "星星": "stars", "房子": "house", "城市": "city",
            "海": "sea", "河": "river", "草": "grass", "鸟": "bird", "鱼": "fish",
            "红色": "red", "蓝色": "blue", "绿色": "green", "黄色": "yellow", "白色": "white",
            "黑色": "black", "大": "big", "小": "small", "美丽": "beautiful", "可爱": "cute",
            "森林": "forest", "沙漠": "desert", "雪": "snow", "雨": "rain", "云": "cloud",
            "汽车": "car", "飞机": "airplane", "船": "boat", "桥": "bridge", "路": "road",
            "眼睛": "eyes", "头发": "hair", "脸": "face", "手": "hand", "微笑": "smile",
            "悲伤": "sad", "快乐": "happy", "愤怒": "angry", "恐惧": "fear",
            "老虎": "tiger", "狮子": "lion", "熊": "bear", "兔子": "rabbit", "马": "horse",
            "龙": "dragon", "凤凰": "phoenix", "蝴蝶": "butterfly", "玫瑰": "rose",
        }
        
        # 1. 加载 CLIP 图像编码器（IP-Adapter 用）
        logger.info("  📦 加载 CLIP 图像编码器...")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=self.dtype
        ).to(self.device)
        
        # 2. 加载双 ControlNet 模型
        logger.info("  📦 加载 Scribble ControlNet...")
        self.controlnet_scribble = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", torch_dtype=self.dtype
        ).to(self.device)
        
        logger.info("  📦 加载 Depth ControlNet...")
        self.controlnet_depth = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=self.dtype
        ).to(self.device)
        
        # 3. 初始化多重 ControlNet 管道
        logger.info("  📦 初始化 Multi-ControlNet 管道...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[self.controlnet_scribble, self.controlnet_depth],  # 双 ControlNet
            image_encoder=self.image_encoder,
            torch_dtype=self.dtype,
            safety_checker=None
        ).to(self.device)

        # 4. 挂载 IP-Adapter
        logger.info("  📦 加载 IP-Adapter...")
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        
        # 5. 一次性加载 LCM-LoRA 并命名（避免重复加载）
        logger.info("  📦 加载 LCM-LoRA (adapter_name='lcm')...")
        self.pipe.load_lora_weights(
            "latent-consistency/lcm-lora-sdv1-5",
            adapter_name="lcm"
        )
        self.lcm_adapter_loaded = True
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # 设置初始适配器（仅 LCM）
        self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
        
        # 6. 占位图资产 (解决 NoneType 报错)
        self.black_placeholder = Image.new('RGB', (224, 224), (0, 0, 0))
        # 深度图占位（全黑代表无深度信息）
        self.black_depth_placeholder = Image.new('RGB', (512, 512), (0, 0, 0))
        # 缓存最后一次处理的深度图（供前端预览）
        self._last_depth_image: Optional[Image.Image] = None
        
        self.style_configs = {
            # ============================================================
            # 风格 LoRA 配置（灵魂注入核心）
            # ============================================================
            # 格式：风格名 -> {prompt, lora_trigger, lora_file}
            # - prompt: 基础风格提示词
            # - lora_trigger: LoRA 专属触发词（必须与下载页面一致！）
            # - lora_file: LoRA 文件名（放在 /root/autodl-tmp/models/loras/ 下）
            #              设为 None 表示该风格暂无 LoRA，降级使用 IP-Adapter
            # ============================================================
            
            # 梵高风格 - 使用 vg.safetensors
            "vangogh": {
                "prompt": "oil painting by Van Gogh, thick impasto brushstrokes, swirling patterns, post-impressionist",
                "lora_trigger": "vg",  # LoRA 触发词
                "lora_file": "vg.safetensors"
            },
            
            # 莫奈风格 - 使用 monet_v2-000004.safetensors
            "monet": {
                "prompt": "Monet style impressionist painting, soft light, water reflections, dreamy atmosphere",
                "lora_trigger": "painting (medium)",  # LoRA 触发词
                "lora_file": "monet_v2-000004.safetensors"
            },
            
            # 油画风格 - 使用 ImpastoBrush LoRa（厚涂笔触）
            "oil": {
                "prompt": "oil painting masterpiece, classical technique, rich textures, thick paint layers",
                "lora_trigger": "impasto",  # LoRA 触发词
                "lora_file": "ImpastoBrush LoRa v1.0.1.safetensors"
            },
            
            # 浮世绘风格 - 使用 Ukiyo-e.safetensors
            "浮世绘": {
                "prompt": "ukiyo-e Japanese woodblock print, bold outlines, flat colors, traditional Japanese art",
                "lora_trigger": "ukiyo-e",  # LoRA 触发词
                "lora_file": "Ukiyo-e.safetensors"
            },
            
            # ============================================================
            # 以下风格暂无 LoRA，使用 IP-Adapter 降级模式
            # ============================================================
            
            "watercolor": {
                "prompt": "watercolor painting, wet-on-wet technique, flowing colors, transparent layers",
                "lora_trigger": "",
                "lora_file": None  # 暂无 LoRA
            },
            
            "sculpture": {
                "prompt": "detailed marble sculpture, classical Greek style, chiaroscuro lighting, museum quality",
                "lora_trigger": "",
                "lora_file": None  # 暂无 LoRA
            },
            
            "cyberpunk": {
                "prompt": "cyberpunk neon city, futuristic, glowing lights, blade runner aesthetic, sci-fi",
                "lora_trigger": "",
                "lora_file": None  # 暂无 LoRA
            },
            
            "赛博朋克": {
                "prompt": "cyberpunk neon city, futuristic, glowing lights, blade runner aesthetic, sci-fi",
                "lora_trigger": "",
                "lora_file": None  # 暂无 LoRA
            }
        }
        
        # 7. 风格素材库：嵌套字典 {风格名: {文件名: PIL对象}}
        self._style_image_cache: Dict[str, Dict[str, Image.Image]] = {}
        # 8. 素材结构缓存：{风格名: [文件名列表]}
        self._assets_structure: Dict[str, List[str]] = {}
        
        self._scan_and_preload_assets()
        logger.info("✅ 4090 全性能引擎已焊死，准备爆发！")

    def _scan_and_preload_assets(self):
        """
        递归扫描风格素材目录，预加载所有图片至显存缓存
        目录结构：/root/autodl-tmp/assets/styles/{风格名}/{图片文件}
        """
        logger.info(f"📂 正在扫描风格素材库: {self.style_assets_dir}")
        
        if not os.path.exists(self.style_assets_dir):
            logger.warning(f"⚠️ 素材目录不存在: {self.style_assets_dir}")
            return
        
        # 遍历风格子目录
        for style_name in os.listdir(self.style_assets_dir):
            if style_name.startswith('.'):
                continue
            style_path = os.path.join(self.style_assets_dir, style_name)
    
            if not os.path.isdir(style_path):
                continue
                
            self._style_image_cache[style_name] = {}
            self._assets_structure[style_name] = []
            
            # 遍历风格文件夹内的图片
            for filename in os.listdir(style_path):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    continue
                    
                file_path = os.path.join(style_path, filename)
                try:
                    # 预加载图片并缓存（保持 float16 以优化显存）
                    pil_img = Image.open(file_path).convert("RGB").resize((224, 224))
                    self._style_image_cache[style_name][filename] = pil_img
                    self._assets_structure[style_name].append(filename)
                    logger.debug(f"  ✅ 加载: {style_name}/{filename}")
                except Exception as e:
                    logger.warning(f"  ❌ 加载失败: {style_name}/{filename} - {e}")
            
            count = len(self._assets_structure[style_name])
            logger.info(f"  📁 {style_name}: {count} 张素材")
        
        total = sum(len(v) for v in self._assets_structure.values())
        logger.info(f"🎨 素材库加载完成，共 {len(self._assets_structure)} 个风格，{total} 张图片")

    def _manage_lora(self, style: str, strength: float = 0.8) -> bool:
        """
        动态 LoRA 加载管理器 - 优化版（避免 14 秒延迟）
        
        核心优化：
        1. LCM-LoRA 在 __init__ 时一次性加载，永不卸载
        2. 风格 LoRA 使用增量加载，仅加载/切换风格适配器
        3. 无 LoRA 时仅将风格适配器权重设为 0，不卸载
        
        返回：
        - True: 成功加载风格 LoRA
        - False: 该风格无 LoRA，降级为纯 IP-Adapter 模式
        """
        # 获取风格配置
        style_config = self.style_configs.get(style)
        
        # 解析 LoRA 配置
        if isinstance(style_config, str):
            lora_file = None
        elif isinstance(style_config, dict):
            lora_file = style_config.get("lora_file")
        else:
            # 未知风格，尝试自动匹配同名 LoRA
            lora_file = f"{style}.safetensors"
            lora_path = os.path.join(self.lora_dir, lora_file)
            if not os.path.exists(lora_path):
                lora_file = None
        
        # ============ 情况1：目标风格无 LoRA ============
        if lora_file is None:
            if self.loaded_style_lora is not None:
                # 有已加载的风格 LoRA，将其权重设为 0（不卸载）
                logger.info(f"🔄 风格 '{style}' 无 LoRA，禁用当前风格 LoRA")
                try:
                    # 仅保留 LCM，禁用风格 LoRA
                    self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
                except Exception as e:
                    logger.warning(f"⚠️ 适配器切换异常: {e}")
                # 注意：不设置 self.loaded_style_lora = None，保留以便快速恢复
            else:
                logger.debug(f"ℹ️ 风格 '{style}' 使用 IP-Adapter 模式（无 LoRA）")
            return False
        
        # ============ 情况2：已加载相同的 LoRA ============
        if self.loaded_style_lora == lora_file:
            logger.debug(f"✅ 风格 LoRA 已加载，仅调整权重: {lora_file}")
            # 直接调整权重，无需重新加载
            try:
                l_scale = float(0.6 + strength * 0.4)
                self.pipe.set_adapters(
                    ["lcm", self.lora_adapter_name],
                    adapter_weights=[1.0, l_scale]
                )
            except Exception as e:
                logger.warning(f"⚠️ LoRA 权重调整失败: {e}")
            return True
        
        # ============ 情况3：需要加载新的风格 LoRA ============
        lora_path = os.path.join(self.lora_dir, lora_file)
        
        # 检查文件是否存在
        if not os.path.exists(lora_path):
            logger.warning(f"⚠️ LoRA 文件不存在: {lora_path}")
            logger.info(f"🎨 降级为 IP-Adapter 模式")
            try:
                self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
            except:
                pass
            return False
        
        # 执行 LoRA 加载
        try:
            logger.info(f"🎭 检测到 LoRA，正在加载...")
            logger.info(f"   风格: {style}, 文件: {lora_file}")
            
            # 如果已有风格 LoRA，需要先删除旧的适配器
            if self.loaded_style_lora is not None:
                try:
                    logger.debug(f"   删除旧适配器: {self.lora_adapter_name}")
                    self.pipe.delete_adapters([self.lora_adapter_name])
                except Exception as e:
                    logger.debug(f"   删除旧适配器时出现警告（可忽略）: {e}")
            
            # 加载新的风格 LoRA（增量加载，不影响 LCM）
            self.pipe.load_lora_weights(
                lora_path,
                adapter_name=self.lora_adapter_name
            )
            
            # 设置双适配器权重
            l_scale = float(0.6 + strength * 0.4)
            self.pipe.set_adapters(
                ["lcm", self.lora_adapter_name],
                adapter_weights=[1.0, l_scale]
            )
            
            self.loaded_style_lora = lora_file
            logger.info(f"✅ 风格 LoRA 加载成功！(耗时极短，无需重载 LCM)")
            return True
            
        except Exception as e:
            logger.error(f"❌ 风格 LoRA 加载失败: {lora_file} - {e}")
            # 回滚：仅使用 LCM
            try:
                self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
            except:
                pass
            self.loaded_style_lora = None
            return False

    def _repair_unet(self):
        """防御性代码：动态对齐接口"""
        if not hasattr(self.pipe.unet, "attn_processors"):
            try: self.pipe.unet.set_attn_processor(self.pipe.unet.get_attn_processor())
            except: pass
    
    def _get_reference_image(self, style: str, ref_image_name: Optional[str] = None) -> Image.Image:
        """
        获取参考图：
        - 若指定 ref_image_name，则从缓存中检索
        - 若未指定，则使用该风格的第一张图
        - 若风格无素材，返回黑色占位图
        """
        style_cache = self._style_image_cache.get(style, {})
        
        if not style_cache:
            logger.warning(f"⚠️ 风格 '{style}' 无可用素材，使用占位图")
            return self.black_placeholder
        
        if ref_image_name and ref_image_name in style_cache:
            logger.debug(f"📷 使用指定参考图: {style}/{ref_image_name}")
            return style_cache[ref_image_name]
        
        # 默认使用第一张图
        first_filename = list(style_cache.keys())[0]
        logger.debug(f"📷 使用默认参考图: {style}/{first_filename}")
        return style_cache[first_filename]

    def _extract_edges_from_photo(self, image: np.ndarray) -> np.ndarray:
        """
        将彩色照片转换为 ControlNet-Scribble 能理解的边缘线条图
        使用自适应阈值 + Canny 边缘检测的组合策略
        """
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 膨胀边缘使线条更粗
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 反转：黑底白线 -> 白底黑线（符合草图习惯）
        edges_inverted = 255 - edges
        
        # 转回 BGR 格式
        edges_bgr = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)
        
        logger.debug("📷 照片边缘提取完成")
        return edges_bgr

    def _extract_depth_from_photo(self, image: np.ndarray) -> Image.Image:
        """
        从照片中提取伪深度图（用于 Depth ControlNet）
        使用基于梯度的轻量级算法，无需额外模型加载
        
        算法原理：
        1. 使用 Sobel 算子提取图像梯度
        2. 结合亮度信息估算深度（较暗区域通常较远）
        3. 高斯模糊平滑深度图
        """
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Sobel 梯度计算（边缘通常是深度变化区域）
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 归一化梯度
        gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # 结合亮度信息：较暗区域设为"较远"（深度值较小）
        # 使用反向亮度作为基础深度
        brightness_depth = 255 - gray
        brightness_depth = cv2.normalize(brightness_depth, None, 0, 255, cv2.NORM_MINMAX)
        
        # 融合梯度和亮度信息
        # 梯度高的地方表示深度边界，亮度用于填充区域
        depth_map = (gradient_norm * 0.4 + brightness_depth * 0.6).astype(np.uint8)
        
        # 高斯模糊使深度图更平滑自然
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
        
        # 增强对比度
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # 转为 RGB 格式的 PIL Image（Depth ControlNet 需要）
        depth_rgb = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        depth_pil = Image.fromarray(depth_rgb)
        
        logger.debug("📷 深度图提取完成")
        return depth_pil

    def _get_depth_image(self, image: np.ndarray, is_photo_mode: bool) -> Image.Image:
        """
        获取深度图：
        - 照片模式：提取伪深度图
        - 草图模式：返回全黑占位图（无深度信息）
        """
        h, w = image.shape[:2]  # 动态获取输入图像的尺寸
        if is_photo_mode:
            return self._extract_depth_from_photo(image)
        else:
            # 草图模式：全黑深度图，动态生成
            return Image.new('RGB', (w, h), (0, 0, 0))

    def _translate_chinese(self, text: str) -> str:
        """
        轻量级中文翻译：使用词表映射 + 保留英文部分
        避免安装额外翻译库，适合 AutoDL 环境
        """
        if not text:
            return text
        
        # 检测是否包含中文
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        if not has_chinese:
            return text
        
        logger.info(f"🔤 检测到中文 Prompt，正在翻译: {text}")
        
        # 使用词表映射翻译
        translated = text
        for cn, en in self.cn_to_en_map.items():
            translated = translated.replace(cn, en)
        
        # 移除剩余的中文字符（无法翻译的部分）
        result = ""
        for char in translated:
            if '\u4e00' <= char <= '\u9fff':
                result += " "  # 用空格替代未翻译的中文
            else:
                result += char
        
        # 清理多余空格
        result = " ".join(result.split())
        
        logger.info(f"🔤 翻译结果: {result}")
        return result

    def process(self, image: np.ndarray, style: str, prompt: str = "", **kwargs) -> np.ndarray:
        """
        高性能推理循环 - 简化版（移除 BLIP 语义识别）
        
        核心架构：
        - 风格 LoRA: 注入大师笔触风格（主导）
        - IP-Adapter: 辅助调色
        - 双 ControlNet: 结构 + 深度控制
        
        返回：
        - result_image: 生成的图像
        """
        start_time = time.time()
        self._repair_unet()
        
        # ============ 显存保护：强制锁死 512 像素 ============
        SAFE_MAX_SIDE = 512
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side > SAFE_MAX_SIDE:
            scale = SAFE_MAX_SIDE / max_side
            new_w = int(w * scale) // 8 * 8 or 8
            new_h = int(h * scale) // 8 * 8 or 8
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"🛡️ 显存保护: 图像从 {w}x{h} 缩放至 {new_w}x{new_h}")

        # 强制类型安全：确保 strength 是 float，防止 int 导致 Diffusers 报错
        strength = float(kwargs.get('strength', 0.6))
        strength = max(0.0, min(1.0, strength))

        is_photo_mode = kwargs.get('is_photo_mode', False)

        # ============ 中文翻译 ============
        translated_prompt = self._translate_chinese(prompt)

        # ============ 动态 LoRA 管理 ============
        has_style_lora = self._manage_lora(style, strength)

        # ============ 预处理：提取 Scribble 边缘图 ============
        if is_photo_mode:
            logger.info("📷 检测到照片模式，正在提取边缘线条...")
            processed_image = self._extract_edges_from_photo(image)
        else:
            processed_image = image

        scribble_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        # ============ 预处理：提取深度图 ============
        depth_image = self._get_depth_image(image, is_photo_mode)
        self._last_depth_image = depth_image

        # 获取参考图
        ref_image_name = kwargs.get('ref_image_name', None)
        ref_img = self._get_reference_image(style, ref_image_name)

        has_valid_ref = (style in self._style_image_cache and 
                         len(self._style_image_cache[style]) > 0)

        # ============ 权重策略 ============
        if has_style_lora:
            # LoRA 权重：与 strength 关联
            l_scale = float(0.6 + strength * 0.4)

            # 动态设置 LoRA 适配器权重
            try:
                self.pipe.set_adapters(
                    ["lcm", self.lora_adapter_name],
                    adapter_weights=[1.0, l_scale]
                )
            except Exception as e:
                logger.warning(f"⚠️ LoRA 权重设置失败: {e}")

            # IP-Adapter 权重
            i_scale = 0.25 if has_valid_ref else 0.0

            logger.info(f"🎭 灵魂注入模式 - LoRA(笔触): {l_scale:.2f}, IP-Adapter(配色): {i_scale:.2f}")
        else:
            l_scale = 0.0
            i_scale = float(0.85 - strength * 0.45) if has_valid_ref else 0.0
            logger.info(f"🎨 IP-Adapter 降级模式 - IP-Adapter: {i_scale:.2f}")

        # ============ ControlNet 权重映射 ============
        if is_photo_mode:
            c_scale = float(0.8 + strength * 0.4)
            d_scale = float(c_scale * 0.5)
        else:
            c_scale = float(0.5 + strength * 0.7)
            d_scale = float(c_scale * 0.1)

        self.pipe.set_ip_adapter_scale(i_scale)

        # ============ Prompt 增强策略 ============
        style_config = self.style_configs.get(style)

        if isinstance(style_config, dict):
            base_style_prompt = style_config.get("prompt", "best quality")
            lora_trigger = style_config.get("lora_trigger", "")
        elif isinstance(style_config, str):
            base_style_prompt = style_config
            lora_trigger = ""
        else:
            base_style_prompt = "best quality, masterpiece"
            lora_trigger = ""

        negative_prompt = "blurry, messy, deformed lines, low quality, watermarks, ugly, distorted, bad anatomy, extra limbs"

        # 构建 LoRA 触发词部分
        lora_prefix = f"({lora_trigger}:1.2), " if has_style_lora and lora_trigger else ""

        # 用户意图部分（已翻译）
        user_intent = f"({translated_prompt}:1.3), " if translated_prompt.strip() else ""

        # 根据模式构建完整提示词
        if is_photo_mode:
            mode_boost = "(photograph style transfer, preserve structure, detailed:1.2), "
            full_prompt = f"{user_intent}{lora_prefix}{mode_boost}{base_style_prompt}"
        elif strength > 0.7:
            mode_boost = "(masterpiece, clean lines, sharp edges:1.2), "
            full_prompt = f"{user_intent}{lora_prefix}{mode_boost}{base_style_prompt}"
        elif strength > 0.4:
            full_prompt = f"{user_intent}{lora_prefix}{base_style_prompt}, detailed, high quality"
        else:
            mode_boost = "(masterpiece, artistic, vibrant colors:1.2), "
            full_prompt = f"{user_intent}{lora_prefix}{mode_boost}{base_style_prompt}"

        logger.info(f"🎨 权重配置 - Scribble: {c_scale:.2f}, Depth: {d_scale:.2f}, IP-Adapter: {i_scale:.2f}, LoRA: {l_scale:.2f}")
        logger.info(f"📝 Prompt: {full_prompt[:100]}..." if len(full_prompt) > 100 else f"📝 Prompt: {full_prompt}")

        with torch.inference_mode(), torch.autocast("cuda"):
            output = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=[scribble_image, depth_image],
                ip_adapter_image=[ref_img],
                num_inference_steps=8,
                guidance_scale=3.5,
                controlnet_conditioning_scale=[c_scale, d_scale],
            )
            result_pil = output[0][0] if isinstance(output, tuple) else output.images[0]
        
        res = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        
        mode_str = "灵魂注入" if has_style_lora else "IP-Adapter"
        logger.info(f"✨ 4090 {mode_str}模式 总处理耗时: {1000*(time.time()-start_time):.2f}ms")
        return res

    def get_last_depth_image_base64(self) -> Optional[str]:
        """
        获取最后一次处理的深度图的 Base64 编码
        用于前端深度图预览
        """
        if self._last_depth_image is None:
            return None
        
        try:
            # 缩小深度图以减少传输数据量
            depth_resized = self._last_depth_image.resize((128, 128), Image.LANCZOS)
            
            # 转为 Base64
            buffer = BytesIO()
            depth_resized.save(buffer, format='JPEG', quality=70)
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_str}"
        except Exception as e:
            logger.warning(f"⚠️ 深度图编码失败: {e}")
            return None

    def get_supported_styles(self) -> List[str]:
        """返回所有支持的风格（包含素材库中发现的风格）"""
        # 合并配置中的风格和素材库中的风格
        all_styles = set(self.style_configs.keys())
        all_styles.update(self._assets_structure.keys())
        return list(all_styles)
    
    def get_assets_structure(self) -> Dict[str, List[str]]:
        """返回风格素材库结构：{风格名: [文件名列表]}"""
        return self._assets_structure


class ClassicNSTProcessor:
    """
    经典 NST 精修版处理器
    调用 Neural-Style-Transfer 项目的 INetwork.py 进行像素级风格迁移。
    遵循 README 与 Guide.md 的黄金法则，针对 4090 显卡进行深度优化。
    """
    def __init__(self):
        # 获取当前文件所在目录作为项目根目录
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        # 核心：使用改进版的 INetwork.py 以获得更高的艺术质量
        self.nst_path = os.path.join(self.project_root, "Neural-Style-Transfer", "INetwork.py")
        self.assets_root = os.path.join(self.project_root, "assets", "styles")
        self._assets_structure = self._scan_assets_directory()
        logger.info("🎨 经典 NST 精修引擎已挂载 (4090 深度优化版)")

    def _scan_assets_directory(self):
        result = {}
        if not os.path.exists(self.assets_root):
            return result
        for style_name in os.listdir(self.assets_root):
            if style_name.startswith('.'): continue
            style_path = os.path.join(self.assets_root, style_name)
            if os.path.isdir(style_path):
                files = [f for f in os.listdir(style_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
                if files:
                    result[style_name] = sorted(files)
        return result

    @staticmethod
    def _original_color_transfer(content_img: np.ndarray, stylized_img: np.ndarray) -> np.ndarray:
        """
        基于 YCrCb 色彩空间的颜色迁移后处理
        保留原图色彩，仅迁移纹理，不会产生条纹伪影
        
        Args:
            content_img: 原始内容图 (BGR)
            stylized_img: 风格化后的图像 (BGR)
        Returns:
            保留原图色彩的风格化图像
        """
        # 确保尺寸一致
        if content_img.shape[:2] != stylized_img.shape[:2]:
            stylized_img = cv2.resize(stylized_img, (content_img.shape[1], content_img.shape[0]), 
                                       interpolation=cv2.INTER_LANCZOS4)
        
        # 转换到 YCrCb 色彩空间
        content_ycrcb = cv2.cvtColor(content_img, cv2.COLOR_BGR2YCrCb)
        stylized_ycrcb = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2YCrCb)
        
        # 用原图的色度通道 (Cr, Cb) 替换风格图的色度通道
        # 保留风格图的亮度通道 (Y) 以保持纹理细节
        stylized_ycrcb[:, :, 1] = content_ycrcb[:, :, 1]  # Cr 通道
        stylized_ycrcb[:, :, 2] = content_ycrcb[:, :, 2]  # Cb 通道
        
        # 转回 BGR
        result = cv2.cvtColor(stylized_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        return result

    def process(self, image: np.ndarray, style: str, prompt: str = "", **kwargs) -> np.ndarray:
        # ============ 强制显存回收：释放主进程 PyTorch 缓存给子进程 TensorFlow 使用 ============
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(0.5)  # 给 GPU 留出物理响应时间
        logger.info("🧹 已强制回收 GPU 显存 (empty_cache + ipc_collect + gc + sleep)")
        
        task_id = str(uuid.uuid4())
        res_prefix = f"/tmp/res_{task_id}"
        content_path = f"/tmp/content_{task_id}.jpg"
        
        # ============ 显存保护：强制锁死 512 像素 ============
        # 回归古早可行版本的计算压力水平，确保 4090 显存绝对安全
        SAFE_MAX_SIDE = 512
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side > SAFE_MAX_SIDE:
            scale = SAFE_MAX_SIDE / max_side
            # VGG16 有 5 个池化层，必须满足 2^5=32 对齐才能保证特征图维度闭合
            new_w = int(w * scale) // 32 * 32 or 32
            new_h = int(h * scale) // 32 * 32 or 32
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"🛡️ 显存保护: 图像从 {w}x{h} 缩放至 {new_w}x{new_h} (32倍数对齐)")
        
        # 1. 动态确定风格参考图路径
        ref_image_name = kwargs.get('ref_image_name')
        if ref_image_name:
            style_path = os.path.join(self.assets_root, style, ref_image_name)
        else:
            style_path = os.path.join(self.assets_root, f"{style}.jpg")
        
        # 高质量保存内容图，避免 JPEG 压缩会引入噪声
        cv2.imwrite(content_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # 2. 【大师级参数配置】严格遵循 README 与 Guide 指南
        # 显存竞争优化：暂时降低迭代次数以减少峰值显存占用
        num_iter = kwargs.get('num_iter', 50)
        
        # 强烈推荐使用 conv5_2 作为内容层以获得更具艺术感的纹理
        content_layer = "conv5_2" 
        
        # 核心建议：严禁使用 noise，必须用 content 初始化以减少噪点
        init_mode = "content"   
        
        # 权重调整：使用 conv5_2 层时，降低风格权重以避免数值溡散
        # 默认采用 1:0.025 的比例，可通过 kwargs 动态调整
        content_weight = 1.0
        style_weight = float(kwargs.get('style_weight', 0.025))
        
        # 池化方式：Max 锐利，Average 柔和（适合流体风格如星空）
        pool_type = kwargs.get('pool_type', 'max')
        
        # 颜色保留模式：保持原图色彩，仅迁移纹理
        preserve_color = kwargs.get('preserve_color', False)
        
        # 总变差权重 (TV Weight)：增加以平滑条纹，避免数值溡散
        tv_weight = 1e-4

        python_bin = "/root/miniconda3/bin/python"
        
        # 动态获取输入图像的尺寸，强制锁死 512 以内
        h, w = image.shape[:2]
        max_dim = max(h, w)
        # VGG16 有 5 个池化层，必须 32 倍数对齐才能保证特征图维度闭合
        image_size = min(max_dim, 512)
        image_size = (image_size // 32) * 32 or 32

        cmd = [
            python_bin, self.nst_path,
            content_path, style_path, res_prefix,
            "--num_iter", str(num_iter),
            "--content_layer", content_layer,
            "--init_image", init_mode,
            "--content_weight", str(content_weight),
            "--style_weight", str(style_weight),
            "--total_variation_weight", str(tv_weight),
            "--image_size", str(image_size),
            "--model", "vgg16",
            "--pool_type", pool_type  # 动态池化方式: max(锐利) 或 ave(柔和)
        ]
        
        logger.info(f"🚀 启动精修模式 | 迭代:{num_iter} | 风格权重:{style_weight} | 池化:{pool_type} | 保色:{preserve_color}")
        
        try:
            custom_env = os.environ.copy()
            custom_env["PYTHONPATH"] = os.path.join(self.project_root, "Neural-Style-Transfer")
            custom_env["CUDA_VISIBLE_DEVICES"] = "0"
            # 关键：限制 TensorFlow 显存分配策略，不贪婪占用
            custom_env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            custom_env["CUDA_MODULE_LOADING"] = "LAZY"
            # 延长超时时间以应对高质量迭代
            process = subprocess.run(cmd, env=custom_env, capture_output=True, text=True, timeout=900)
            
            if process.returncode != 0:
                logger.error(f"❌ 渲染失败: {process.stderr}")
                return image
        except Exception as e:
            logger.error(f"❌ 系统异常: {e}")
            return image
        
        # 3. 读取最终结果
        final_output_path = f"{res_prefix}_at_iteration_{num_iter}.png"
        if os.path.exists(final_output_path):
            res = cv2.imread(final_output_path)
            if res is not None:
                # 如果开启了颜色保留模式，使用 OpenCV 后处理进行色彩迁移
                if preserve_color:
                    logger.info("🎨 应用色彩保留后处理 (YCrCb 色彩空间迁移)")
                    res = self._original_color_transfer(image, res)
                return res
            return image
        return image

    def get_supported_styles(self):
        return list(self._assets_structure.keys())
    
    def get_assets_structure(self):
        return self._assets_structure

def create_processor(processor_type="ai", **kwargs):
    if processor_type.lower() == "ai":
        return AIProcessor()
    elif processor_type.lower() == "nst":
        return ClassicNSTProcessor()
    else:
        return MockProcessor()