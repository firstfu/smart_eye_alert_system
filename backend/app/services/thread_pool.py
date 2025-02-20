import threading
import queue
import concurrent.futures
from typing import Callable, Any, Dict, Optional
import logging
import time
from dataclasses import dataclass
from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """任務資訊"""
    func: Callable  # 要執行的函數
    args: tuple     # 位置參數
    kwargs: dict    # 關鍵字參數
    priority: int   # 優先級（數字越小優先級越高）
    timestamp: float  # 建立時間戳

class ThreadPoolManager:
    """執行緒池管理器"""
    def __init__(self, max_workers: int = None):
        """初始化執行緒池管理器

        Args:
            max_workers: 最大工作執行緒數，預設為 CPU 核心數 * 2
        """
        self.max_workers = max_workers or (settings.CPU_CORES * 2)
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="worker"
        )
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # 效能監控
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        self._stats_lock = threading.Lock()

    def submit_task(self, func: Callable, *args, priority: int = 1, **kwargs) -> str:
        """提交任務到執行緒池

        Args:
            func: 要執行的函數
            *args: 位置參數
            priority: 優先級（1-5，1最高）
            **kwargs: 關鍵字參數

        Returns:
            str: 任務ID
        """
        task_id = str(time.time())
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=max(1, min(5, priority)),  # 確保優先級在1-5之間
            timestamp=time.time()
        )

        # 將任務加入優先級佇列
        self.task_queue.put((task.priority, task_id, task))
        logger.debug(f"已提交任務 {task_id}，優先級 {task.priority}")
        return task_id

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """獲取任務結果

        Args:
            task_id: 任務ID
            timeout: 超時時間（秒）

        Returns:
            Any: 任務結果

        Raises:
            TimeoutError: 等待超時
            KeyError: 任務不存在
        """
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            if task_id in self.results:
                result = self.results[task_id]
                del self.results[task_id]  # 清理結果
                return result
            time.sleep(0.1)

        raise TimeoutError(f"等待任務 {task_id} 結果超時")

    def _worker_loop(self):
        """工作執行緒主迴圈"""
        while self.running:
            try:
                # 從佇列取出任務
                priority, task_id, task = self.task_queue.get(timeout=1.0)

                # 提交任務到執行緒池
                start_time = time.time()
                future = self.executor.submit(task.func, *task.args, **task.kwargs)

                try:
                    # 等待任務完成
                    result = future.result()
                    self.results[task_id] = result

                    # 更新統計資訊
                    processing_time = time.time() - start_time
                    with self._stats_lock:
                        self.stats["tasks_completed"] += 1
                        self.stats["total_processing_time"] += processing_time
                        self.stats["avg_processing_time"] = (
                            self.stats["total_processing_time"] /
                            self.stats["tasks_completed"]
                        )

                except Exception as e:
                    logger.error(f"任務 {task_id} 執行失敗: {str(e)}")
                    with self._stats_lock:
                        self.stats["tasks_failed"] += 1
                    self.results[task_id] = e

                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"工作執行緒迴圈發生錯誤: {str(e)}")

    def get_stats(self) -> Dict:
        """獲取執行緒池統計資訊

        Returns:
            Dict: 統計資訊
        """
        with self._stats_lock:
            return {
                "max_workers": self.max_workers,
                "active_threads": len([t for t in threading.enumerate()
                                    if t.name.startswith("worker")]),
                "queue_size": self.task_queue.qsize(),
                "tasks_completed": self.stats["tasks_completed"],
                "tasks_failed": self.stats["tasks_failed"],
                "avg_processing_time": self.stats["avg_processing_time"]
            }

    def shutdown(self):
        """關閉執行緒池"""
        self.running = False
        self.worker_thread.join()
        self.executor.shutdown(wait=True)
        logger.info("執行緒池已關閉")