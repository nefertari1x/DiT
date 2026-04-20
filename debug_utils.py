"""
调试工具：监控每层的激活值、梯度、参数，定位 loss spike 的来源。

用法：在训练脚本中：
    from debug_utils import GradientMonitor
    monitor = GradientMonitor(model, rank=rank, log_dir="results/xxx/debug")

    # 训练循环中：
    loss.backward()
    anomaly = monitor.check_after_backward(train_steps, loss.item())
    if anomaly:
        monitor.dump_snapshot(train_steps, x, t, y, depth_seq, depth_centers)
        # 可选：跳过这一步更新
        opt.zero_grad()
        continue
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
"""

import torch
import torch.nn as nn
import json
import os
from collections import deque


class GradientMonitor:
    def __init__(self, model, rank=0, log_dir="debug_logs",
                 window_size=100, spike_threshold=5.0):
        """
        Args:
            model: DDP wrapped model
            rank: 当前进程 rank，只在 rank=0 时写文件
            log_dir: 调试日志输出目录
            window_size: 用最近 N 步的 loss 均值作为基准
            spike_threshold: loss 超过均值的倍数时判定为 spike
        """
        self.model = model
        self.rank = rank
        self.log_dir = log_dir
        self.window_size = window_size
        self.spike_threshold = spike_threshold

        # loss 历史（滑动窗口）
        self.loss_history = deque(maxlen=window_size)

        # 注册 hooks 来捕获每层的激活值和梯度统计
        self.activation_stats = {}  # forward 激活值统计
        self.grad_stats = {}        # backward 梯度统计（对激活值的梯度）
        self._hooks = []

        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)

        self._register_hooks()

    def _register_hooks(self):
        """给模型的每个关键层注册 forward/backward hook。"""
        # 获取底层模型（去掉 DDP 和 compile 包装）
        base = self.model
        if hasattr(base, 'module'):
            base = base.module
        if hasattr(base, '_orig_mod'):
            base = base._orig_mod

        # 监控每个 DiT block
        self._adaln_clamp_sources = []  # 记录有 clamp 的 block，用于读取 post-clamp 值
        for i, block in enumerate(base.blocks):
            name = f"block_{i}"
            self._register_one(block, name)
            # 更细粒度：分别监控 attention 和 MLP
            self._register_one(block.attn, f"block_{i}.attn")
            self._register_one(block.mlp, f"block_{i}.mlp")
            self._register_one(block.adaLN_modulation, f"block_{i}.adaLN")
            if getattr(block, 'adaln_clamp', 0) > 0:
                self._adaln_clamp_sources.append((f"block_{i}.adaLN_clamped", block))

        # 监控其他关键组件
        self._register_one(base.x_embedder, "x_embedder")
        self._register_one(base.t_embedder, "t_embedder")
        self._register_one(base.y_embedder, "y_embedder")
        self._register_one(base.final_layer, "final_layer")
        if getattr(base.final_layer, 'adaln_clamp', 0) > 0:
            self._adaln_clamp_sources.append(("final_layer.adaLN_clamped", base.final_layer))

        if getattr(base, 'use_depth', False):
            self._register_one(base.depth_value_embed, "depth_value_embed")
            self._register_one(base.depth_k_proj, "depth_k_proj")
            self._register_one(base.depth_v_proj, "depth_v_proj")
            for i, block in enumerate(base.blocks):
                if getattr(block, 'use_cross', False):
                    self._register_one(block.cross_attn, f"block_{i}.cross_attn")

    def _register_one(self, module, name):
        """给单个 module 注册 forward 和 backward hook。"""
        def fwd_hook(mod, inp, out, _name=name):
            if isinstance(out, torch.Tensor):
                self.activation_stats[_name] = self._tensor_stats(out)
            elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                self.activation_stats[_name] = self._tensor_stats(out[0])

        def bwd_hook(mod, grad_in, grad_out, _name=name):
            # grad_out 是 loss 对该层输出的梯度
            if isinstance(grad_out, tuple) and grad_out[0] is not None:
                self.grad_stats[_name] = self._tensor_stats(grad_out[0])
            elif isinstance(grad_out, torch.Tensor):
                self.grad_stats[_name] = self._tensor_stats(grad_out)

        h1 = module.register_forward_hook(fwd_hook)
        h2 = module.register_full_backward_hook(bwd_hook)
        self._hooks.append(h1)
        self._hooks.append(h2)

    @staticmethod
    def _tensor_stats(t):
        """计算一个 tensor 的关键统计量（在 detach 后，不影响计算图）。"""
        with torch.no_grad():
            t_float = t.detach().float()  # 转 FP32 以避免 BF16 精度问题
            return {
                "mean": t_float.mean().item(),
                "std": t_float.std().item(),
                "min": t_float.min().item(),
                "max": t_float.max().item(),
                "abs_max": t_float.abs().max().item(),
                "has_nan": bool(torch.isnan(t_float).any()),
                "has_inf": bool(torch.isinf(t_float).any()),
                "nan_frac": torch.isnan(t_float).float().mean().item(),
            }

    def check_after_backward(self, step, loss_val):
        """
        在 loss.backward() 之后调用。
        返回 True 表示检测到异常，建议跳过这一步。
        """
        self._collect_post_clamp_stats()

        # 更新 loss 历史
        self.loss_history.append(loss_val)

        # 判断是否有异常
        anomaly_reasons = []

        # 1. loss 本身是 NaN/Inf
        if not (loss_val == loss_val) or abs(loss_val) == float('inf'):
            anomaly_reasons.append(f"loss is {loss_val}")

        # 2. loss spike 检测
        if len(self.loss_history) >= 10:
            recent = list(self.loss_history)[-self.window_size:]
            mean_loss = sum(recent) / len(recent)
            if mean_loss > 0 and loss_val > mean_loss * self.spike_threshold:
                anomaly_reasons.append(
                    f"loss spike: {loss_val:.6f} > {self.spike_threshold}x mean({mean_loss:.6f})"
                )

        # 3. 检查激活值中是否有 NaN/Inf
        for name, stats in self.activation_stats.items():
            if stats["has_nan"]:
                anomaly_reasons.append(f"activation NaN in {name}")
            if stats["has_inf"]:
                anomaly_reasons.append(f"activation Inf in {name}")

        # 4. 检查梯度中是否有 NaN/Inf
        for name, stats in self.grad_stats.items():
            if stats["has_nan"]:
                anomaly_reasons.append(f"gradient NaN in {name}")
            if stats["has_inf"]:
                anomaly_reasons.append(f"gradient Inf in {name}")

        # 5. 检查每层参数的梯度
        param_grad_anomalies = self._check_param_grads()
        anomaly_reasons.extend(param_grad_anomalies)

        if anomaly_reasons and self.rank == 0:
            self._log_anomaly(step, loss_val, anomaly_reasons)

        return len(anomaly_reasons) > 0

    def _check_param_grads(self):
        """检查模型参数的梯度是否有 NaN/Inf。"""
        anomalies = []
        base = self.model
        if hasattr(base, 'module'):
            base = base.module

        for name, param in base.named_parameters():
            if param.grad is not None:
                g = param.grad
                if torch.isnan(g).any():
                    anomalies.append(f"param grad NaN: {name}")
                if torch.isinf(g).any():
                    anomalies.append(f"param grad Inf: {name}")
        return anomalies

    def _log_anomaly(self, step, loss_val, reasons):
        """将异常信息写入日志文件。"""
        record = {
            "step": step,
            "loss": loss_val,
            "reasons": reasons,
            "activation_stats": self.activation_stats,
            "gradient_stats": self.grad_stats,
        }

        # 写入按 step 命名的 JSON 文件
        path = os.path.join(self.log_dir, f"anomaly_step{step:07d}.json")
        with open(path, 'w') as f:
            json.dump(record, f, indent=2, default=str)

        # 也打印到 stdout
        print(f"\n{'='*60}")
        print(f"[ANOMALY] Step {step}, Loss={loss_val:.6f}")
        print(f"Reasons: {reasons}")
        print(f"--- Activation stats (sorted by abs_max) ---")
        sorted_act = sorted(self.activation_stats.items(),
                            key=lambda x: x[1]["abs_max"], reverse=True)
        for name, s in sorted_act[:15]:
            flag = " *** NaN!" if s["has_nan"] else (" *** Inf!" if s["has_inf"] else "")
            print(f"  {name:30s}  abs_max={s['abs_max']:12.4f}  "
                  f"mean={s['mean']:10.4f}  std={s['std']:10.4f}{flag}")
        print(f"--- Gradient stats (sorted by abs_max) ---")
        sorted_grad = sorted(self.grad_stats.items(),
                             key=lambda x: x[1]["abs_max"], reverse=True)
        for name, s in sorted_grad[:15]:
            flag = " *** NaN!" if s["has_nan"] else (" *** Inf!" if s["has_inf"] else "")
            print(f"  {name:30s}  abs_max={s['abs_max']:12.4f}  "
                  f"mean={s['mean']:10.4f}  std={s['std']:10.4f}{flag}")
        print(f"{'='*60}\n")

    def dump_snapshot(self, step, x, t, y, depth_seq=None, depth_centers=None):
        """
        保存触发异常的 batch 数据，便于离线复现。
        """
        if self.rank != 0:
            return
        path = os.path.join(self.log_dir, f"batch_step{step:07d}.pt")
        torch.save({
            "step": step,
            "x": x.detach().cpu(),
            "t": t.detach().cpu(),
            "y": y.detach().cpu(),
            "depth_seq": depth_seq.detach().cpu() if depth_seq is not None else None,
            "depth_centers": depth_centers.detach().cpu() if depth_centers is not None else None,
        }, path)
        print(f"[DEBUG] Saved anomaly batch to {path}")

    def _collect_post_clamp_stats(self):
        """读取各 block/final_layer 的 _adaln_post_clamp 属性，计算统计量。"""
        for name, module in self._adaln_clamp_sources:
            t = getattr(module, '_adaln_post_clamp', None)
            if t is not None:
                self.activation_stats[name] = self._tensor_stats(t)

    def log_periodic(self, step, tb_writer=None):
        """
        每 N 步调用一次（比如每 log_every 步），记录正常状态下的统计量到 TensorBoard。
        用于观察趋势——比如某层的 abs_max 是否在缓慢增长。
        """
        if self.rank != 0:
            return

        self._collect_post_clamp_stats()

        if tb_writer is not None:
            for name, stats in self.activation_stats.items():
                tb_writer.add_scalar(f"debug_act/{name}_absmax", stats["abs_max"], step)
            for name, stats in self.grad_stats.items():
                tb_writer.add_scalar(f"debug_grad/{name}_absmax", stats["abs_max"], step)

            # 参数本身的统计量
            base = self.model
            if hasattr(base, 'module'):
                base = base.module
            for pname, param in base.named_parameters():
                if param.grad is not None:
                    gnorm = param.grad.detach().float().norm().item()
                    tb_writer.add_scalar(f"debug_pgrad/{pname}", gnorm, step)

    def remove_hooks(self):
        """移除所有 hooks。"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
