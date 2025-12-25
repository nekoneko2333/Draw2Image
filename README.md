# Draw2Image 🎨

**人机交互技术大作业** - 实时草图/照片艺术风格转换系统

一个基于深度学习的实时图像风格迁移系统，支持草图实时生成艺术画作和照片风格转换。采用 Stable Diffusion + ControlNet + LoRA + IP-Adapter 多模态融合架构，结合经典神经风格迁移算法，提供从速览到精修的完整创作工作流。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ 功能特性

### 🖌️ 实时草图生成
- **WebSocket 实时通信**：笔触即时反馈，毫秒级响应
- **多风格支持**：梵高、莫奈、浮世绘、油画、水彩、赛博朋克等 12+ 种艺术风格
- **LoRA 笔触注入**：加载预训练的艺术大师笔触风格模型
- **IP-Adapter 配色参考**：使用参考图进行智能配色
- **双 ControlNet 控制**：
  - Scribble ControlNet：保持线条结构
  - Depth ControlNet：增强空间深度感

### 📷 照片风格转换
- **自动边缘提取**：从照片中智能提取线条轮廓
- **深度图估算**：基于梯度和亮度的伪深度图生成
- **结构保持**：在风格迁移的同时保留照片主体结构

### 🎭 经典神经风格迁移（精修模式）
- 基于 VGG16/VGG19 的传统 NST 算法
- 高质量迭代优化，适合最终精修输出
- 支持颜色保留、多种池化策略
- 批量处理支持，可配合预设快速生成

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端界面 (index.html)                      │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│    │   画布绘制    │    │   风格选择    │    │   参数调节    │     │
│    └──────────────┘    └──────────────┘    └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI 后端 (main.py)                       │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│    │  WebSocket   │    │  REST API    │    │  静态资源     │     │
│    │   /ws        │    │  /api/*      │    │  /static     │     │
│    └──────────────┘    └──────────────┘    └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    处理器层 (processor.py)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    AIProcessor                           │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐ │    │
│  │  │ SD v1.5   │ │ControlNet│ │ IP-Adapter│ │  LoRA    │ │    │
│  │  │           │ │ Scribble │ │           │ │ 艺术风格  │ │    │
│  │  │           │ │  Depth   │ │           │ │          │ │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └──────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               ClassicNSTProcessor                        │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │    Neural-Style-Transfer (VGG16/19)               │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                MockProcessor (降级方案)                   │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │    │
│  │  │ 油画 │ │ 素描 │ │ 水彩 │ │ 卡通 │ │ 浮雕 │ │ 铅笔 │       │    │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
Draw2Image/
├── main.py                 # FastAPI 主入口，WebSocket 服务
├── processor.py            # 核心处理器（AI/NST/Mock）
├── batch_nst.py            # 批量神经风格迁移脚本
├── environment.yml         # Conda 环境配置
├── README.md               # 项目文档
│
├── static/                 # 前端静态资源
│   ├── index.html          # 实时草图生成界面
│   └── refined.html        # 精修模式界面
│
├── assets/                 # 风格素材库
│   └── styles/             # 风格参考图片
│       ├── vangogh/        # 梵高风格
│       ├── monet/          # 莫奈风格
│       ├── oil/            # 油画风格
│       ├── sculpture/      # 雕塑风格
│       ├── 浮世绘/          # 浮世绘风格
│       ├── 中国水墨/        # 中国水墨风格
│       ├── 赛博朋克/        # 赛博朋克风格
│       ├── 巴洛克/          # 巴洛克风格
│       ├── 文艺复兴/        # 文艺复兴风格
│       ├── 立体主义/        # 立体主义风格
│       ├── 超现实主义/      # 超现实主义风格
│       └── 平面插画/        # 平面插画风格
│
├── models/                 # 预训练模型
│   └── loras/              # LoRA 模型文件
│       ├── vg.safetensors                      # 梵高风格 LoRA
│       ├── monet_v2-000004.safetensors         # 莫奈风格 LoRA
│       ├── Ukiyo-e.safetensors                 # 浮世绘风格 LoRA
│       └── ImpastoBrush LoRa v1.0.1.safetensors # 厚涂油画 LoRA
│
├── Neural-Style-Transfer/  # 经典 NST 模块
│   ├── INetwork.py         # 改进版 NST 算法
│   ├── utils.py            # 工具函数
│   ├── Guide.md            # 参数调优指南
│   ├── README.md           # NST 模块文档
│   └── images/             # 示例图片
│
├── input/                  # 批量处理输入目录
├── ref/                    # 批量处理风格参考目录
└── output/                 # 批量处理输出目录
```

---

## 🚀 快速开始

### 环境要求

- **Python** 3.8+
- **CUDA** 11.7+ (推荐 GPU 运行)
- **显存** ≥ 8GB (推荐 12GB+)
- **Conda** (推荐使用 Anaconda/Miniconda)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/nekoneko2333/Draw2Image.git
cd Draw2Image
```

2. **创建 Conda 环境**
```bash
conda env create -f environment.yml
conda activate base  # 或您的环境名
```

3. **安装核心依赖**
```bash
pip install fastapi uvicorn websockets
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install opencv-python pillow numpy
```

4. **下载预训练模型**

模型会在首次运行时自动从 HuggingFace 下载：
- `runwayml/stable-diffusion-v1-5`
- `lllyasviel/sd-controlnet-scribble`
- `lllyasviel/sd-controlnet-depth`
- `h94/IP-Adapter`
- `latent-consistency/lcm-lora-sdv1-5`

> 💡 **提示**: 中国大陆用户建议配置 HuggingFace 镜像：
> ```python
> os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
> ```

5. **启动服务**
```bash
python main.py
```

6. **访问界面**

打开浏览器访问 `http://localhost:8000/static/index.html`

---

## 📖 使用指南

### 实时草图模式

1. 在左侧画布上绘制草图
2. 从右侧面板选择艺术风格
3. 可选择具体的风格参考图
4. 调节「结构保留」滑块控制风格强度
5. 在 Prompt 输入框描述画面内容（支持中文）
6. 实时查看右侧生成结果

### 照片转换模式

1. 点击「上传照片」按钮
2. 系统自动提取边缘和深度信息
3. 选择目标艺术风格
4. 调整参数后自动生成

### 精修模式

访问 `/static/refined.html` 使用经典 NST 算法进行高质量精修：
- 更长的迭代时间换取更精细的效果
- 支持颜色保留选项
- 支持不同池化策略（Max/Average）

### 批量处理

```bash
# 基础用法
python batch_nst.py

# 使用预设模式
python batch_nst.py --preset balanced  # fast/balanced/quality/ultra

# 自定义参数
python batch_nst.py --num_iter 200 --style_weight 0.05 --image_size 768

# 筛选特定文件
python batch_nst.py --content "cat" --style "monet" --limit 10
```

**预设模式说明**：

| 预设 | 迭代次数 | 图像尺寸 | 预计耗时 | 适用场景 |
|------|----------|----------|----------|----------|
| fast | 50 | 400px | 1-2 分钟 | 快速预览 |
| balanced | 100 | 512px | 3-5 分钟 | 日常使用 |
| quality | 200 | 768px | 8-15 分钟 | 高质量输出 |
| ultra | 500 | 1024px | 20-40 分钟 | 极致品质 |

---

## 🎨 支持的艺术风格

### LoRA 风格（真实笔触）

| 风格 | 触发词 | 说明 |
|------|--------|------|
| 梵高 (vangogh) | `vg` | 厚重的后印象派笔触，漩涡纹理 |
| 莫奈 (monet) | `painting (medium)` | 印象派光影，梦幻氛围 |
| 油画 (oil) | `impasto` | 经典厚涂技法，丰富纹理 |
| 浮世绘 | `ukiyo-e` | 日本木版画，平面色块 |

### 纯 IP-Adapter 风格

| 风格 | 特点 |
|------|------|
| 水彩 (watercolor) | 湿染技法，透明层次 |
| 雕塑 (sculpture) | 大理石质感，古典光影 |
| 赛博朋克 | 霓虹光效，未来科幻 |
| 中国水墨 | 传统笔墨，留白意境 |
| 巴洛克 | 华丽装饰，戏剧光影 |
| 文艺复兴 | 古典构图，细腻写实 |
| 立体主义 | 几何分割，多视角 |
| 超现实主义 | 梦境奇幻，超越现实 |
| 平面插画 | 现代设计，扁平风格 |

---

## ⚙️ API 接口

### WebSocket 实时生成

**端点**: `ws://localhost:8000/ws`

**请求格式**:
```json
{
  "image": "data:image/png;base64,...",
  "style": "vangogh",
  "prompt": "一只可爱的猫",
  "strength": 0.6,
  "ref_image_name": "starry_night.jpg",
  "is_photo_mode": false
}
```

**响应格式**:
```json
{
  "type": "result",
  "image": "data:image/jpeg;base64,...",
  "depth": "data:image/jpeg;base64,..."
}
```

### REST API

**获取素材结构**:
```
GET /api/assets
```

**精修渲染**:
```
POST /api/render_refined
Content-Type: application/json

{
  "image": "base64...",
  "style": "vangogh",
  "ref_image_name": "reference.jpg",
  "preserve_color": false,
  "pool_type": "max",
  "style_weight": 0.05
}
```

---

## 🔧 参数说明

### 处理器参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strength` | float | 0.6 | 结构保留程度 (0.0-1.0)，越高保留越多原始结构 |
| `style` | string | "vangogh" | 目标风格名称 |
| `prompt` | string | "" | 画面描述（支持中文自动翻译） |
| `ref_image_name` | string | null | 指定的参考图文件名 |
| `is_photo_mode` | bool | false | 是否为照片模式 |

### NST 批量处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_iter` | 100 | 优化迭代次数 |
| `--image_size` | 512 | 输出图像最大边长 |
| `--content_weight` | 1.0 | 内容损失权重 |
| `--style_weight` | 0.05 | 风格损失权重 |
| `--content_layer` | conv5_2 | 内容特征提取层 |
| `--pool_type` | max | 池化类型 (max/ave) |
| `--preserve_color` | false | 保留原图颜色 |
| `--tv_weight` | 8.5e-5 | 总变差正则化权重 |

---

## 🛠️ 技术栈

### 核心框架
- **FastAPI** - 高性能异步 Web 框架
- **PyTorch** - 深度学习框架
- **Diffusers** - Stable Diffusion 推理库

### AI 模型
- **Stable Diffusion v1.5** - 基础生成模型
- **ControlNet** - 条件控制网络
- **IP-Adapter** - 图像提示适配器
- **LCM-LoRA** - 快速推理加速
- **VGG16/19** - 特征提取网络

### 图像处理
- **OpenCV** - 图像处理
- **PIL/Pillow** - 图像 I/O
- **NumPy** - 数值计算

---

## 📊 性能优化

### GPU 内存管理
- 动态图像缩放保护 (最大 512px)
- LoRA 热切换，避免重复加载
- 显存自动回收 (`torch.cuda.empty_cache()`)

### 推理加速
- LCM-LoRA 8 步快速推理
- FP16 混合精度计算
- 条件编译的 ControlNet 权重

### 资源预加载
- 启动时扫描并缓存所有风格素材
- PIL Image 预转换为 GPU 可用格式

---

## 🐛 常见问题

### Q: 显存不足怎么办？

A: 尝试以下方案：
1. 降低输入图像尺寸
2. 使用 MockProcessor 降级方案
3. 减少 ControlNet 权重
4. 关闭照片模式的深度图

### Q: 生成速度慢？

A: 确保：
1. 使用 GPU 运行
2. LCM-LoRA 正确加载
3. 推理步数设为 8（已默认）

### Q: 中文 Prompt 不生效？

A: 系统内置简单中英词表翻译，复杂描述建议直接使用英文。

### Q: 风格效果不明显？

A: 调整 `strength` 参数（降低值可增强风格），或选择有 LoRA 支持的风格。

---

## 📝 更新日志

### v1.2.0
- ✨ 新增照片模式自动边缘提取
- ✨ 新增深度图估算和预览
- 🎨 增加多种中文风格支持
- ⚡ 优化 LoRA 热切换性能

### v1.1.0
- ✨ 集成 IP-Adapter 配色参考
- ✨ 支持双 ControlNet
- 🔧 批量处理脚本增强

### v1.0.0
- 🎉 初始版本发布
- ✨ 实时 WebSocket 草图生成
- ✨ 经典 NST 精修模式

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

Neural-Style-Transfer 模块基于 [titu1994/Neural-Style-Transfer](https://github.com/titu1994/Neural-Style-Transfer)

---

## 🙏 致谢

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [LCM-LoRA](https://github.com/luosiallen/latent-consistency-model)
- [Neural-Style-Transfer](https://github.com/titu1994/Neural-Style-Transfer)

---

<p align="center">
  Made with ❤️ for 人机交互技术课程
</p>
