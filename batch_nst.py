"""
批量神经风格迁移脚本
读取 input/ 目录下的内容图，使用 ref/ 目录下的风格参考图，
依次调用 INetwork.py 进行风格迁移，结果保存到 output/ 目录。

用法：
    python batch_nst.py
    python batch_nst.py --num_iter 100 --style_weight 0.5
"""

import os
import sys
import subprocess
import argparse
import random
from datetime import datetime

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 目录配置
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
REF_DIR = os.path.join(PROJECT_ROOT, "ref")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# INetwork.py 路径
INETWORK_PATH = os.path.join(PROJECT_ROOT, "Neural-Style-Transfer", "INetwork.py")
NST_DIR = os.path.join(PROJECT_ROOT, "Neural-Style-Transfer")

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def get_image_files(directory):
    """获取目录下所有图片文件，按文件名排序"""
    if not os.path.exists(directory):
        return []
    files = []
    for f in os.listdir(directory):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(f)
    return sorted(files)


def run_nst(content_path, style_path, output_prefix, **kwargs):
    """
    调用 INetwork.py 进行单次风格迁移
    
    Args:
        content_path: 内容图路径
        style_path: 风格参考图路径
        output_prefix: 输出文件前缀（不含扩展名）
        **kwargs: 传递给 INetwork.py 的额外参数
    
    Returns:
        bool: 是否成功
    """
    # 构建命令
    cmd = [
        sys.executable,  # 使用当前 Python 解释器
        INETWORK_PATH,
        content_path,
        style_path,
        output_prefix,
    ]
    
    # 添加可选参数
    if 'num_iter' in kwargs:
        cmd.extend(["--num_iter", str(kwargs['num_iter'])])
    if 'image_size' in kwargs:
        cmd.extend(["--image_size", str(kwargs['image_size'])])
    if 'content_weight' in kwargs:
        cmd.extend(["--content_weight", str(kwargs['content_weight'])])
    if 'style_weight' in kwargs:
        cmd.extend(["--style_weight", str(kwargs['style_weight'])])
    if 'content_layer' in kwargs:
        cmd.extend(["--content_layer", str(kwargs['content_layer'])])
    if 'init_image' in kwargs:
        cmd.extend(["--init_image", str(kwargs['init_image'])])
    if 'pool_type' in kwargs:
        cmd.extend(["--pool_type", str(kwargs['pool_type'])])
    if 'preserve_color' in kwargs:
        cmd.extend(["--preserve_color", str(kwargs['preserve_color'])])
    if 'model' in kwargs:
        cmd.extend(["--model", str(kwargs['model'])])
    if 'tv_weight' in kwargs:
        cmd.extend(["--total_variation_weight", str(kwargs['tv_weight'])])
    if 'save_every' in kwargs:
        cmd.extend(["--save_every", str(kwargs['save_every'])])
    
    print(f"\n{'='*60}")
    print(f"🎨 开始风格迁移")
    print(f"   内容图: {os.path.basename(content_path)}")
    print(f"   风格图: {os.path.basename(style_path)}")
    print(f"   输出: {output_prefix}")
    print(f"{'='*60}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = NST_DIR
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    try:
        process = subprocess.run(
            cmd,
            env=env,
            capture_output=False,  # 直接输出到控制台
            timeout=1800  # 30分钟超时
        )
        return process.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ 超时：处理时间超过30分钟")
        return False
    except Exception as e:
        print(f"❌ 错误：{e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='批量神经风格迁移')
    
    # 预设模式
    parser.add_argument('--preset', type=str, default=None,
                        choices=['fast', 'balanced', 'quality', 'ultra'],
                        help='预设模式: fast(快速预览), balanced(平衡), quality(高质量), ultra(极致)')
    
    parser.add_argument('--num_iter', type=int, default=100,
                        help='迭代次数 (默认: 100，Guide建议100即可)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='输出图像最大边长 (默认: 512)')
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='内容权重 (默认: 1.0)')
    parser.add_argument('--style_weight', type=float, default=0.05,
                        help='风格权重 (默认: 0.05，Guide建议conv5_2时用0.1/0.05/0.01)')
    parser.add_argument('--content_layer', type=str, default='conv5_2',
                        help='内容层 (默认: conv5_2，Guide强烈推荐)')
    parser.add_argument('--init_image', type=str, default='content',
                        choices=['content', 'noise', 'gray'],
                        help='初始化方式 (默认: content，必须用content)')
    parser.add_argument('--pool_type', type=str, default='max',
                        choices=['max', 'ave'],
                        help='池化类型: max(锐利，推荐) 或 ave(柔和) (默认: max)')
    parser.add_argument('--preserve_color', action='store_true',
                        help='保留原图颜色')
    parser.add_argument('--model', type=str, default='vgg16',
                        choices=['vgg16', 'vgg19'],
                        help='VGG模型 (默认: vgg16)')
    parser.add_argument('--tv_weight', type=float, default=8.5e-5,
                        help='总变差权重 (默认: 8.5e-5，Guide说90%%情况适用)')
    parser.add_argument('--pair', action='store_true',
                        help='配对模式：input和ref中同名文件配对处理')
    
    # 筛选参数
    parser.add_argument('--content', type=str, default=None,
                        help='只处理指定的内容图（文件名，支持部分匹配）')
    parser.add_argument('--style', type=str, default=None,
                        help='只使用指定的风格图（文件名，支持部分匹配）')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制最大任务数量')
    parser.add_argument('--random', type=int, default=None,
                        help='每张内容图随机选择N张风格图（默认：使用全部）')
    parser.add_argument('--save_every', type=int, default=10,
                        help='每隔N次迭代保存中间结果（默认: 10，设为0只保存最终结果）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子，用于复现结果')
    
    args = parser.parse_args()
    
    # 预设模式覆盖参数（基于 Guide.md 官方建议）
    PRESETS = {
        'fast': {      # 快速预览，约1-2分钟/张
            'num_iter': 50,
            'image_size': 400,
            'model': 'vgg16',
            'content_layer': 'conv5_2',
            'style_weight': 0.1,
            'tv_weight': 8.5e-5,
        },
        'balanced': {  # 平衡模式，约3-5分钟/张（Guide推荐配置）
            'num_iter': 100,
            'image_size': 512,
            'model': 'vgg16',
            'content_layer': 'conv5_2',
            'style_weight': 0.05,
            'tv_weight': 8.5e-5,
        },
        'quality': {   # 高质量，约8-15分钟/张
            'num_iter': 200,
            'image_size': 768,
            'model': 'vgg16',
            'content_layer': 'conv5_2',
            'style_weight': 0.05,
            'tv_weight': 5e-5,
        },
        'ultra': {     # 极致品质，约20-40分钟/张
            'num_iter': 500,
            'image_size': 1024,
            'model': 'vgg16',
            'content_layer': 'conv5_2',
            'style_weight': 0.025,
            'tv_weight': 1e-5,
        },
    }
    
    if args.preset:
        preset = PRESETS[args.preset]
        print(f"\n🎯 使用预设模式: {args.preset.upper()}")
        # 预设值覆盖默认值（但命令行显式指定的参数优先）
        for key, value in preset.items():
            if key == 'num_iter' and args.num_iter == 100:
                args.num_iter = value
            elif key == 'image_size' and args.image_size == 512:
                args.image_size = value
            elif key == 'model' and args.model == 'vgg16':
                args.model = value
            elif key == 'content_layer' and args.content_layer == 'conv5_2':
                args.content_layer = value
            elif key == 'style_weight' and args.style_weight == 0.05:
                args.style_weight = value
            elif key == 'tv_weight' and args.tv_weight == 8.5e-5:
                args.tv_weight = value
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取文件列表
    input_files = get_image_files(INPUT_DIR)
    ref_files = get_image_files(REF_DIR)
    
    # 筛选内容图
    if args.content:
        input_files = [f for f in input_files if args.content.lower() in f.lower()]
        if not input_files:
            print(f"❌ 错误：未找到匹配 '{args.content}' 的内容图")
            return
    
    # 筛选风格图
    if args.style:
        ref_files = [f for f in ref_files if args.style.lower() in f.lower()]
        if not ref_files:
            print(f"❌ 错误：未找到匹配 '{args.style}' 的风格图")
            return
    
    if not input_files:
        print(f"❌ 错误：input/ 目录为空或不存在")
        print(f"   请将内容图放入: {INPUT_DIR}")
        return
    
    if not ref_files:
        print(f"❌ 错误：ref/ 目录为空或不存在")
        print(f"   请将风格参考图放入: {REF_DIR}")
        return
    
    print(f"\n🖼️  发现 {len(input_files)} 张内容图")
    print(f"🎨 发现 {len(ref_files)} 张风格参考图")
    
    # NST 参数
    nst_kwargs = {
        'num_iter': args.num_iter,
        'image_size': args.image_size,
        'content_weight': args.content_weight,
        'style_weight': args.style_weight,
        'content_layer': args.content_layer,
        'init_image': args.init_image,
        'pool_type': args.pool_type,
        'preserve_color': 'True' if args.preserve_color else 'False',
        'model': args.model,
        'tv_weight': args.tv_weight,
        'save_every': args.save_every,
    }
    
    print(f"\n📋 参数配置:")
    print(f"   迭代次数: {args.num_iter}")
    print(f"   图像尺寸: {args.image_size}")
    print(f"   内容权重: {args.content_weight}")
    print(f"   风格权重: {args.style_weight}")
    print(f"   模型: {args.model}")
    print(f"   保存频率: 每{args.save_every}次迭代" if args.save_every > 0 else "   保存频率: 仅保存最终结果")
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        print(f"   随机种子: {args.seed}")
    
    success_count = 0
    fail_count = 0
    
    if args.pair:
        # 配对模式：同名文件配对
        print(f"\n📌 配对模式：匹配同名文件")
        input_dict = {os.path.splitext(f)[0]: f for f in input_files}
        ref_dict = {os.path.splitext(f)[0]: f for f in ref_files}
        
        common_names = set(input_dict.keys()) & set(ref_dict.keys())
        if not common_names:
            print(f"❌ 错误：未找到同名配对文件")
            return
        
        for name in sorted(common_names):
            content_path = os.path.join(INPUT_DIR, input_dict[name])
            style_path = os.path.join(REF_DIR, ref_dict[name])
            output_prefix = os.path.join(OUTPUT_DIR, f"{name}_styled")
            
            if run_nst(content_path, style_path, output_prefix, **nst_kwargs):
                success_count += 1
            else:
                fail_count += 1
    else:
        # 随机/笛卡尔积模式
        if args.random:
            # 随机模式：每张内容图随机选N张风格图
            total_tasks = len(input_files) * min(args.random, len(ref_files))
            print(f"\n📌 随机模式：每张内容图随机选 {args.random} 张风格图")
        else:
            # 笛卡尔积模式：每张内容图 x 每张风格图
            total_tasks = len(input_files) * len(ref_files)
            print(f"\n📌 笛卡尔积模式")
        
        # 限制任务数量
        if args.limit and total_tasks > args.limit:
            print(f"⚠️ 任务数 {total_tasks} 超过限制 {args.limit}，将只处理前 {args.limit} 个")
            total_tasks = args.limit
        
        print(f"📊 共 {total_tasks} 个任务")
        
        # 估算时间
        est_time_per_task = 3.5  # 分钟
        est_total_time = total_tasks * est_time_per_task
        print(f"⏱️ 预计耗时: {est_total_time:.0f} 分钟 ({est_total_time/60:.1f} 小时)")
        
        task_num = 0
        for content_file in input_files:
            content_name = os.path.splitext(content_file)[0]
            content_path = os.path.join(INPUT_DIR, content_file)
            
            # 确定该内容图使用的风格图列表
            if args.random:
                # 随机选择 N 张风格图
                n = min(args.random, len(ref_files))
                selected_styles = random.sample(ref_files, n)
            else:
                selected_styles = ref_files
            
            for style_file in selected_styles:
                task_num += 1
                
                # 限制任务数量
                if args.limit and task_num > args.limit:
                    break
                
                style_name = os.path.splitext(style_file)[0]
                style_path = os.path.join(REF_DIR, style_file)
                
                # 输出文件名格式: 内容名_风格名_styled
                output_prefix = os.path.join(OUTPUT_DIR, f"{content_name}_{style_name}_styled")
                
                print(f"\n[{task_num}/{total_tasks}]")
                
                if run_nst(content_path, style_path, output_prefix, **nst_kwargs):
                    success_count += 1
                else:
                    fail_count += 1
            
            # 限制任务数量 - 外层循环也要跳出
            if args.limit and task_num >= args.limit:
                break
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"✅ 完成！成功: {success_count}, 失败: {fail_count}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
