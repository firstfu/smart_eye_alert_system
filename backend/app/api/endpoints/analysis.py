from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime

from ...services.llm_analyzer import LLMAnalyzer, AnalysisResult, AnalysisType
from ...core.config import settings

router = APIRouter()
llm_analyzer = LLMAnalyzer()

class SceneAnalysisRequest(BaseModel):
    """場景分析請求"""
    camera_id: str
    scene_description: str
    detected_objects: List[str]
    motion_info: Dict
    quality_info: Dict

class BehaviorAnalysisRequest(BaseModel):
    """行為分析請求"""
    camera_id: str
    behavior_description: str
    motion_history: List[Dict]
    context: Dict

class AnalysisResponse(BaseModel):
    """分析回應"""
    camera_id: str
    analysis_type: str
    confidence: float
    description: str
    details: Dict
    timestamp: float
    requires_action: bool

@router.post("/scene", response_model=AnalysisResponse)
async def analyze_scene(request: SceneAnalysisRequest):
    """分析場景"""
    try:
        result = await llm_analyzer.analyze_scene(
            request.scene_description,
            request.detected_objects,
            request.motion_info,
            request.quality_info
        )

        return AnalysisResponse(
            camera_id=request.camera_id,
            analysis_type=result.type.value,
            confidence=result.confidence,
            description=result.description,
            details=result.details,
            timestamp=result.timestamp,
            requires_action=result.requires_action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/behavior", response_model=AnalysisResponse)
async def analyze_behavior(request: BehaviorAnalysisRequest):
    """分析行為"""
    try:
        result = await llm_analyzer.analyze_behavior(
            request.behavior_description,
            request.motion_history,
            request.context
        )

        return AnalysisResponse(
            camera_id=request.camera_id,
            analysis_type=result.type.value,
            confidence=result.confidence,
            description=result.description,
            details=result.details,
            timestamp=result.timestamp,
            requires_action=result.requires_action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/{camera_id}", response_model=Optional[AnalysisResponse])
async def get_cached_analysis(camera_id: str):
    """獲取快取的分析結果"""
    result = llm_analyzer.get_cached_analysis(camera_id)
    if result:
        return AnalysisResponse(
            camera_id=camera_id,
            analysis_type=result.type.value,
            confidence=result.confidence,
            description=result.description,
            details=result.details,
            timestamp=result.timestamp,
            requires_action=result.requires_action
        )
    return None

@router.delete("/cache")
async def clear_analysis_cache():
    """清除分析快取"""
    llm_analyzer.clear_cache()
    return {"message": "快取已清除"}