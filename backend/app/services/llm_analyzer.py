from typing import Dict, Optional, List
import numpy as np
import cv2
import time
import asyncio
from dataclasses import dataclass
import logging
from enum import Enum
import json
import aiohttp
from ..core.config import settings

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """分析類型"""
    SCENE_UNDERSTANDING = "scene_understanding"  # 場景理解
    BEHAVIOR_ANALYSIS = "behavior_analysis"      # 行為分析
    RISK_ASSESSMENT = "risk_assessment"          # 風險評估
    ANOMALY_DETECTION = "anomaly_detection"      # 異常檢測

@dataclass
class AnalysisResult:
    """分析結果"""
    type: AnalysisType
    confidence: float
    description: str
    details: Dict
    timestamp: float
    requires_action: bool

class LLMAnalyzer:
    """LLM 分析器"""
    def __init__(self):
        """初始化 LLM 分析器"""
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.LLM_MODEL
        self.api_base = settings.OPENAI_API_BASE
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        self.last_call_timestamp = 0
        self.rate_limit_delay = 1.0  # 最小呼叫間隔（秒）

    async def _call_llm_api(self, prompt: str) -> Dict:
        """呼叫 LLM API

        Args:
            prompt: 提示文字

        Returns:
            Dict: API 回應
        """
        # 限制呼叫頻率
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_timestamp
        if time_since_last_call < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last_call)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一個專業的影像分析助手，專門協助分析監控影像中的場景、行為和潛在風險。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    self.last_call_timestamp = time.time()

                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"API 呼叫失敗: {error_text}")

                    return await response.json()

        except Exception as e:
            logger.error(f"LLM API 呼叫失敗: {str(e)}")
            raise

    def _prepare_scene_prompt(self, scene_description: str, detected_objects: List[str],
                            motion_info: Dict, quality_info: Dict) -> str:
        """準備場景分析提示

        Args:
            scene_description: 場景描述
            detected_objects: 偵測到的物件列表
            motion_info: 移動資訊
            quality_info: 影像品質資訊

        Returns:
            str: 提示文字
        """
        return f"""請分析以下監控場景並提供專業評估：

場景描述：{scene_description}

偵測到的物件：{', '.join(detected_objects)}

移動資訊：
- 移動區域：{motion_info.get('location', 'unknown')}
- 移動程度：{motion_info.get('motion_ratio', 0):.2%}

影像品質：
- 亮度：{quality_info.get('brightness', 0):.1f}
- 對比度：{quality_info.get('contrast', 0):.1f}
- 清晰度：{quality_info.get('blur_score', 0):.1f}

請提供：
1. 場景狀況評估
2. 潛在風險分析
3. 建議採取的行動
4. 風險等級（低/中/高）"""

    async def analyze_scene(self,
                          scene_description: str,
                          detected_objects: List[str],
                          motion_info: Dict,
                          quality_info: Dict) -> AnalysisResult:
        """分析場景

        Args:
            scene_description: 場景描述
            detected_objects: 偵測到的物件列表
            motion_info: 移動資訊
            quality_info: 影像品質資訊

        Returns:
            AnalysisResult: 分析結果
        """
        # 生成快取金鑰
        cache_key = f"{scene_description}_{','.join(detected_objects)}_{json.dumps(motion_info)}"

        # 檢查快取
        if cache_key in self.analysis_cache:
            cached_result = self.analysis_cache[cache_key]
            if time.time() - cached_result.timestamp < settings.LLM_CACHE_TTL:
                return cached_result

        # 準備提示文字
        prompt = self._prepare_scene_prompt(
            scene_description,
            detected_objects,
            motion_info,
            quality_info
        )

        # 呼叫 LLM API
        response = await self._call_llm_api(prompt)

        # 解析回應
        try:
            content = response['choices'][0]['message']['content']

            # 簡單的回應解析（實際應用中可能需要更複雜的解析邏輯）
            requires_action = "高" in content or "立即" in content or "緊急" in content
            confidence = 0.8 if "確定" in content or "明確" in content else 0.6

            result = AnalysisResult(
                type=AnalysisType.SCENE_UNDERSTANDING,
                confidence=confidence,
                description=content,
                details={
                    "detected_objects": detected_objects,
                    "motion_info": motion_info,
                    "quality_info": quality_info
                },
                timestamp=time.time(),
                requires_action=requires_action
            )

            # 更新快取
            self.analysis_cache[cache_key] = result

            # 如果快取太大，移除最舊的項目
            if len(self.analysis_cache) > settings.LLM_CACHE_SIZE:
                oldest_key = min(self.analysis_cache.keys(),
                               key=lambda k: self.analysis_cache[k].timestamp)
                del self.analysis_cache[oldest_key]

            return result

        except Exception as e:
            logger.error(f"LLM 回應解析失敗: {str(e)}")
            raise

    async def analyze_behavior(self,
                             behavior_description: str,
                             motion_history: List[Dict],
                             context: Dict) -> AnalysisResult:
        """分析行為

        Args:
            behavior_description: 行為描述
            motion_history: 移動歷史記錄
            context: 上下文資訊

        Returns:
            AnalysisResult: 分析結果
        """
        # 準備提示文字
        prompt = f"""請分析以下行為並評估其風險：

行為描述：{behavior_description}

移動歷史：
{json.dumps(motion_history, indent=2)}

上下文資訊：
{json.dumps(context, indent=2)}

請提供：
1. 行為模式分析
2. 異常行為評估
3. 風險等級判定
4. 建議處置方式"""

        # 呼叫 LLM API
        response = await self._call_llm_api(prompt)

        # 解析回應
        content = response['choices'][0]['message']['content']
        requires_action = "異常" in content or "風險" in content
        confidence = 0.7  # 行為分析的預設信心度

        return AnalysisResult(
            type=AnalysisType.BEHAVIOR_ANALYSIS,
            confidence=confidence,
            description=content,
            details={
                "motion_history": motion_history,
                "context": context
            },
            timestamp=time.time(),
            requires_action=requires_action
        )

    def get_cached_analysis(self, scene_key: str) -> Optional[AnalysisResult]:
        """獲取快取的分析結果

        Args:
            scene_key: 場景金鑰

        Returns:
            Optional[AnalysisResult]: 快取的分析結果
        """
        if scene_key in self.analysis_cache:
            result = self.analysis_cache[scene_key]
            if time.time() - result.timestamp < settings.LLM_CACHE_TTL:
                return result
            else:
                del self.analysis_cache[scene_key]
        return None

    def clear_cache(self):
        """清除分析快取"""
        self.analysis_cache.clear()