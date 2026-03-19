"""
效率监控模块：用于量化训练和推理的计算成本
支持指标：
1. FLOPs (浮点运算次数)
2. 每个epoch的训练时间
3. 每个epoch的推理时间
4. 总训练时间
5. 总推理时间
6. 推理延迟 (p50, p90, p99)
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path


class EfficiencyMonitor:
    """效率监控器"""
    
    def __init__(self, output_dir: str, device: str = 'cuda'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # 时间记录
        self.total_train_start = None
        self.total_train_time = 0.0
        self.epoch_train_times = []
        self.epoch_eval_times = []
        self.total_eval_time = 0.0
        
        # 推理延迟记录 (每个batch的延迟)
        self.inference_latencies = []
        
        # FLOPs记录
        self.flops_computed = False
        self.train_flops = None
        self.inference_flops = None
        
        # 当前epoch计时器
        self.current_epoch_train_start = None
        self.current_epoch_eval_start = None
        
    def start_total_training(self):
        """开始总训练计时"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.total_train_start = time.time()
        
    def end_total_training(self):
        """结束总训练计时"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        if self.total_train_start is not None:
            self.total_train_time = time.time() - self.total_train_start
            
    def start_epoch_training(self):
        """开始epoch训练计时"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.current_epoch_train_start = time.time()
        
    def end_epoch_training(self):
        """结束epoch训练计时"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        if self.current_epoch_train_start is not None:
            epoch_time = time.time() - self.current_epoch_train_start
            self.epoch_train_times.append(epoch_time)
            return epoch_time
        return 0.0
        
    def start_epoch_eval(self):
        """开始epoch评估计时"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.current_epoch_eval_start = time.time()
        
    def end_epoch_eval(self):
        """结束epoch评估计时"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        if self.current_epoch_eval_start is not None:
            eval_time = time.time() - self.current_epoch_eval_start
            self.epoch_eval_times.append(eval_time)
            self.total_eval_time += eval_time
            return eval_time
        return 0.0
        
    def record_inference_latency(self, batch_size: int):
        """记录单个batch的推理延迟"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        # 这个方法应该在推理前后被调用
        # 实际使用时需要配合 start/end 使用
        pass
        
    def measure_batch_latency(self, batch_size: int) -> float:
        """测量并记录batch延迟（毫秒）"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        return start  # 返回开始时间，需要在推理后调用end
        
    def end_batch_latency(self, start_time: float, batch_size: int):
        """结束batch延迟测量"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        latency_ms = (time.time() - start_time) * 1000  # 转换为毫秒
        per_sample_latency = latency_ms / batch_size if batch_size > 0 else latency_ms
        self.inference_latencies.append(per_sample_latency)
        
    def compute_flops(self, model, sample_input, mode='train'):
        """
        计算模型FLOPs
        需要安装: pip install fvcore
        """
        try:
            from fvcore.nn import FlopCountAnalysis
            
            model.eval()
            with torch.no_grad():
                flops = FlopCountAnalysis(model, sample_input)
                total_flops = flops.total()
                
            if mode == 'train':
                # 训练时FLOPs约为推理的3倍（forward + backward）
                self.train_flops = total_flops * 3
                self.inference_flops = total_flops
            else:
                self.inference_flops = total_flops
                
            self.flops_computed = True
            return total_flops
            
        except ImportError:
            print("Warning: fvcore not installed. FLOPs calculation skipped.")
            print("Install with: pip install fvcore")
            return None
        except Exception as e:
            print(f"Warning: FLOPs calculation failed: {e}")
            return None
            
    def get_statistics(self) -> Dict:
        """获取所有统计信息"""
        stats = {
            'total_training_time_hours': self.total_train_time / 3600,
            'total_training_time_seconds': self.total_train_time,
            'total_eval_time_seconds': self.total_eval_time,
            'num_epochs': len(self.epoch_train_times),
            'epoch_train_times_seconds': self.epoch_train_times,
            'epoch_eval_times_seconds': self.epoch_eval_times,
            'avg_epoch_train_time_seconds': np.mean(self.epoch_train_times) if self.epoch_train_times else 0,
            'avg_epoch_eval_time_seconds': np.mean(self.epoch_eval_times) if self.epoch_eval_times else 0,
        }
        
        # 推理延迟统计
        if self.inference_latencies:
            latencies = np.array(self.inference_latencies)
            stats['inference_latency_ms'] = {
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies)),
                'p50': float(np.percentile(latencies, 50)),
                'p90': float(np.percentile(latencies, 90)),
                'p99': float(np.percentile(latencies, 99)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies)),
            }
            
        # FLOPs统计
        if self.flops_computed:
            stats['flops'] = {
                'train_flops': self.train_flops,
                'inference_flops': self.inference_flops,
                'train_gflops': self.train_flops / 1e9 if self.train_flops else None,
                'inference_gflops': self.inference_flops / 1e9 if self.inference_flops else None,
            }
            
        return stats
        
    def print_summary(self):
        """打印效率统计摘要"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("EFFICIENCY STATISTICS SUMMARY")
        print("="*60)
        
        print(f"\n[Training Time]")
        print(f"  Total training time: {stats['total_training_time_hours']:.2f} hours ({stats['total_training_time_seconds']:.1f}s)")
        print(f"  Number of epochs: {stats['num_epochs']}")
        if stats['avg_epoch_train_time_seconds'] > 0:
            print(f"  Avg time per epoch: {stats['avg_epoch_train_time_seconds']:.1f}s")
            
        print(f"\n[Evaluation Time]")
        print(f"  Total eval time: {stats['total_eval_time_seconds']:.1f}s")
        if stats['avg_epoch_eval_time_seconds'] > 0:
            print(f"  Avg eval time per epoch: {stats['avg_epoch_eval_time_seconds']:.1f}s")
            
        if 'inference_latency_ms' in stats:
            lat = stats['inference_latency_ms']
            print(f"\n[Inference Latency (per sample)]")
            print(f"  Mean: {lat['mean']:.2f} ms")
            print(f"  P50:  {lat['p50']:.2f} ms")
            print(f"  P90:  {lat['p90']:.2f} ms")
            print(f"  P99:  {lat['p99']:.2f} ms")
            
        if 'flops' in stats and stats['flops']['train_gflops']:
            flops = stats['flops']
            print(f"\n[FLOPs]")
            print(f"  Training FLOPs: {flops['train_gflops']:.2f} GFLOPs")
            print(f"  Inference FLOPs: {flops['inference_gflops']:.2f} GFLOPs")
            
        print("="*60 + "\n")
        
    def save_to_file(self, filename: str = 'efficiency_stats.json'):
        """保存统计信息到文件"""
        stats = self.get_statistics()
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"Efficiency statistics saved to: {output_path}")
        
    def save_to_wandb(self, run, prefix: str = 'efficiency'):
        """保存统计信息到wandb"""
        if run is None:
            return
            
        stats = self.get_statistics()
        
        # 扁平化字典用于wandb
        wandb_stats = {
            f'{prefix}/total_train_time_hours': stats['total_training_time_hours'],
            f'{prefix}/total_eval_time_seconds': stats['total_eval_time_seconds'],
            f'{prefix}/avg_epoch_train_time': stats['avg_epoch_train_time_seconds'],
            f'{prefix}/avg_epoch_eval_time': stats['avg_epoch_eval_time_seconds'],
        }
        
        if 'inference_latency_ms' in stats:
            lat = stats['inference_latency_ms']
            wandb_stats.update({
                f'{prefix}/latency_mean_ms': lat['mean'],
                f'{prefix}/latency_p50_ms': lat['p50'],
                f'{prefix}/latency_p90_ms': lat['p90'],
                f'{prefix}/latency_p99_ms': lat['p99'],
            })
            
        if 'flops' in stats and stats['flops']['train_gflops']:
            flops = stats['flops']
            wandb_stats.update({
                f'{prefix}/train_gflops': flops['train_gflops'],
                f'{prefix}/inference_gflops': flops['inference_gflops'],
            })
            
        run.log(wandb_stats)
