# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
from torchvision.datasets.utils import download_url
import torch.distributed as dist
from collections import OrderedDict
import torch
import os


pretrained_models = {'DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt'}


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        return download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
    
        # if "model" in checkpoint: 
        #     print("正在加载主模型 (Non-EMA)...")
        #     checkpoint = checkpoint["model"]
        # elif "ema" in checkpoint:
        #     print("主模型未找到，加载 EMA...")
        #     checkpoint = checkpoint["ema"]
            
        return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, 'pretrained_models')
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model

def resume_from_checkpoint(args, model, ema, opt, device, logger, steps_per_epoch=None):
    """
    从 Checkpoint 恢复模型状态，包含针对 torch.compile 的智能 Key 匹配逻辑。
    """
    start_epoch = 0
    train_steps = 0

    if not args.resume:
        return start_epoch, train_steps

    if dist.get_rank() == 0:
        logger.info(f"Resuming checkpoint from: {args.resume}")

    # [重要] weights_only=False 是必须的，因为你的 checkpoint 里包含 'args' (Pickle 对象)
    map_location = f"cuda:{device}"
    checkpoint = torch.load(args.resume, map_location=map_location, weights_only=False)

    # =========================================================
    # 1. 核心修复: 智能匹配模型 Key
    # =========================================================
    # 获取当前运行模型期望的所有 Keys
    if hasattr(model, 'module'):
        target_obj = model.module
    else:
        target_obj = model
    
    target_keys = set(target_obj.state_dict().keys())
    
    # 原始 Checkpoint 权重
    raw_state_dict = checkpoint["model"]
    new_state_dict = OrderedDict()
    
    # 统计匹配情况，用于 Debug
    match_stats = {
        "exact": 0,          # 直接匹配
        "add_prefix": 0,     # 添加 _orig_mod. 后匹配 (你的情况)
        "remove_prefix": 0,  # 移除 _orig_mod. 后匹配
        "missing": 0         # 无法匹配
    }

    for k, v in raw_state_dict.items():
        # 策略 A: 精确匹配 (最理想情况)
        if k in target_keys:
            new_state_dict[k] = v
            match_stats["exact"] += 1
            continue
            
        # 策略 B: Checkpoint 是干净的，但模型是 Compiled (缺 _orig_mod.)
        # 尝试添加前缀看是否存在于目标中
        k_with_prefix = f"_orig_mod.{k}"
        if k_with_prefix in target_keys:
            new_state_dict[k_with_prefix] = v
            match_stats["add_prefix"] += 1
            continue
            
        # 策略 C: Checkpoint 是 Compiled，但模型是干净的 (多 _orig_mod.)
        # 尝试移除前缀看是否存在于目标中
        if k.startswith("_orig_mod."):
            k_no_prefix = k.replace("_orig_mod.", "")
            if k_no_prefix in target_keys:
                new_state_dict[k_no_prefix] = v
                match_stats["remove_prefix"] += 1
                continue
        
        # 策略 D: 实在找不到，保留原样 (可能会报 Missing key 错误)
        new_state_dict[k] = v
        match_stats["missing"] += 1

    # 打印匹配统计信息
    if dist.get_rank() == 0:
        logger.info(f"Weight Loading Stats: Exact={match_stats['exact']}, "
                    f"Added Prefix(_orig_mod.)={match_stats['add_prefix']}, "
                    f"Removed Prefix={match_stats['remove_prefix']}, "
                    f"Missing/Unknown={match_stats['missing']}")

    # 加载适配后的权重
    target_obj.load_state_dict(new_state_dict)

    # =========================================================
    # 2. 加载 EMA 和 优化器
    # =========================================================
    # EMA 通常是不编译的，所以我们通常需要保证 EMA 权重是“干净”的
    if ema is not None and "ema" in checkpoint:
        raw_ema_state = checkpoint["ema"]
        new_ema_dict = OrderedDict()
        ema_target_keys = set(ema.state_dict().keys())
        
        for k, v in raw_ema_state.items():
            # 同样的适配逻辑用于 EMA，但 EMA 通常没有 prefix
            if k in ema_target_keys:
                new_ema_dict[k] = v
                continue
            
            # 尝试去前缀 (常见情况: 如果以前存 EMA 不小心带了前缀)
            if k.startswith("_orig_mod."):
                k_no_prefix = k.replace("_orig_mod.", "")
                if k_no_prefix in ema_target_keys:
                    new_ema_dict[k_no_prefix] = v
                    continue
                    
            new_ema_dict[k] = v
            
        ema.load_state_dict(new_ema_dict)
    
    if opt is not None and "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])

    # =========================================================
    # 3. 恢复训练进度 (Steps & Epoch)
    # =========================================================
    if "train_steps" in checkpoint:
        train_steps = checkpoint["train_steps"]
    else:
        try:
            fname = os.path.basename(args.resume)
            train_steps = int(fname.split('.')[0])
        except ValueError:
            if dist.get_rank() == 0:
                logger.warning("Could not extract train_steps. Starting from 0.")
            train_steps = 0

    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"] + 1
    elif steps_per_epoch is not None and steps_per_epoch > 0:
        start_epoch = train_steps // steps_per_epoch
        if dist.get_rank() == 0:
            logger.info(f"Inferring start_epoch={start_epoch} from train_steps={train_steps}")

    if dist.get_rank() == 0:
        logger.info(f"Resumed successfully at Epoch {start_epoch}, Step {train_steps}")

    return start_epoch, train_steps

if __name__ == "__main__":
    # Download all DiT checkpoints
    for model in pretrained_models:
        download_model(model)
    print('Done.')
