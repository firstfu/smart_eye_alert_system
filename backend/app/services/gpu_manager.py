import torch
import torch.cuda as cuda
import numpy as np
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
import threading
from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """GPU 資訊"""
    device_id: int
    name: str
    total_memory: int  # MB
    used_memory: int   # MB
    temperature: float
    utilization: float  # %

class GPUManager:
    """GPU 資源管理器"""
    def __init__(self):
        """初始化 GPU 資源管理器"""
        self.available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.available else 0
        self.current_device = 0
        self._lock = threading.Lock()

        if self.available:
            logger.info(f"找到 {self.device_count} 個 GPU 設備")
            for i in range(self.device_count):
                device_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {device_name}")
        else:
            logger.warning("未找到可用的 GPU 設備")

    def get_optimal_device(self) -> torch.device:
        """獲取最佳的 GPU 設備

        Returns:
            torch.device: GPU 設備，如果沒有可用的 GPU 則返回 CPU
        """
        if not self.available or not settings.ENABLE_GPU:
            return torch.device('cpu')

        with self._lock:
            # 獲取所有 GPU 的使用情況
            gpu_stats = self.get_gpu_stats()
            if not gpu_stats:
                return torch.device('cpu')

            # 選擇記憶體使用率最低的 GPU
            optimal_gpu = min(gpu_stats, key=lambda x: x.used_memory)
            return torch.device(f'cuda:{optimal_gpu.device_id}')

    def get_gpu_stats(self) -> List[GPUInfo]:
        """獲取所有 GPU 的狀態

        Returns:
            List[GPUInfo]: GPU 狀態列表
        """
        if not self.available:
            return []

        gpu_stats = []
        for i in range(self.device_count):
            try:
                # 獲取記憶體資訊
                memory = torch.cuda.get_device_properties(i).total_memory
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_cached = torch.cuda.memory_reserved(i)

                # 轉換為 MB
                total_memory = memory / 1024 / 1024
                used_memory = (memory_allocated + memory_cached) / 1024 / 1024

                # 獲取設備名稱
                device_name = torch.cuda.get_device_name(i)

                # 獲取溫度和使用率（如果可用）
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except:
                    temperature = 0.0
                    utilization = 0.0

                gpu_stats.append(GPUInfo(
                    device_id=i,
                    name=device_name,
                    total_memory=int(total_memory),
                    used_memory=int(used_memory),
                    temperature=temperature,
                    utilization=utilization
                ))

            except Exception as e:
                logger.error(f"獲取 GPU {i} 狀態失敗: {str(e)}")
                continue

        return gpu_stats

    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """優化張量的記憶體使用

        Args:
            tensor: 輸入張量

        Returns:
            torch.Tensor: 優化後的張量
        """
        if not tensor.is_cuda:
            return tensor

        # 使用記憶體釋放器
        torch.cuda.empty_cache()

        # 如果張量不需要梯度，則分離它
        if tensor.requires_grad:
            tensor = tensor.detach()

        return tensor

    def batch_to_gpu(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """將一批數據轉移到 GPU

        Args:
            batch: 包含張量的字典

        Returns:
            Dict[str, torch.Tensor]: 轉移到 GPU 的數據
        """
        if not self.available or not settings.ENABLE_GPU:
            return batch

        device = self.get_optimal_device()
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def preprocess_for_gpu(self, image: np.ndarray) -> torch.Tensor:
        """預處理影像以在 GPU 上使用

        Args:
            image: 輸入影像

        Returns:
            torch.Tensor: 處理後的張量
        """
        # 轉換為 float32 並正規化
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        # 轉換為 PyTorch 張量
        tensor = torch.from_numpy(image)

        # 如果是 3 通道影像，調整維度順序
        if len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1)

        # 增加批次維度
        tensor = tensor.unsqueeze(0)

        # 移動到最佳 GPU
        device = self.get_optimal_device()
        tensor = tensor.to(device)

        return tensor

    def postprocess_from_gpu(self, tensor: torch.Tensor) -> np.ndarray:
        """將 GPU 張量轉換回 CPU numpy 陣列

        Args:
            tensor: GPU 上的張量

        Returns:
            np.ndarray: CPU 上的 numpy 陣列
        """
        # 確保張量在 CPU 上
        tensor = tensor.cpu()

        # 如果有批次維度，移除它
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # 如果需要，調整通道順序
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)

        # 轉換為 numpy 並還原值範圍
        array = tensor.numpy()
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)

        return array

    def cleanup(self):
        """清理 GPU 資源"""
        if self.available:
            torch.cuda.empty_cache()
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except:
                pass