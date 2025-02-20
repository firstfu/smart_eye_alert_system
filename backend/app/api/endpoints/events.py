from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from ...services.event_processor import Event, EventType
from ...services.stream_manager import StreamManager
from ...core.config import settings

router = APIRouter()
stream_manager = StreamManager()

@router.get("/events")
async def list_events(
    camera_id: Optional[str] = None,
    event_type: Optional[EventType] = None,
    start_time: Optional[float] = Query(None, description="起始時間戳"),
    end_time: Optional[float] = Query(None, description="結束時間戳"),
    limit: int = Query(100, ge=1, le=1000, description="最大返回數量")
) -> List[dict]:
    """獲取事件列表

    Args:
        camera_id: 攝影機ID過濾
        event_type: 事件類型過濾
        start_time: 起始時間戳
        end_time: 結束時間戳
        limit: 最大返回數量

    Returns:
        List[dict]: 事件列表
    """
    events = stream_manager.get_events(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time
    )

    if event_type:
        events = [e for e in events if e.type == event_type]

    # 按時間戳排序（最新的在前）
    events.sort(key=lambda x: x.timestamp, reverse=True)

    # 限制返回數量
    events = events[:limit]

    # 轉換為 dict 格式
    return [
        {
            "type": e.type.value,
            "level": e.level.value,
            "camera_id": e.camera_id,
            "timestamp": e.timestamp,
            "details": e.details,
            "frame_timestamp": e.frame_timestamp
        }
        for e in events
    ]

@router.get("/events/types")
async def list_event_types() -> List[str]:
    """獲取所有事件類型

    Returns:
        List[str]: 事件類型列表
    """
    return [e.value for e in EventType]

@router.get("/events/stats")
async def get_event_stats(
    camera_id: Optional[str] = None,
    start_time: Optional[float] = Query(None, description="起始時間戳"),
    end_time: Optional[float] = Query(None, description="結束時間戳")
) -> dict:
    """獲取事件統計資訊

    Args:
        camera_id: 攝影機ID過濾
        start_time: 起始時間戳
        end_time: 結束時間戳

    Returns:
        dict: 統計資訊
    """
    events = stream_manager.get_events(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time
    )

    # 計算各類型事件數量
    type_counts = {}
    for event in events:
        event_type = event.type.value
        type_counts[event_type] = type_counts.get(event_type, 0) + 1

    # 計算各等級事件數量
    level_counts = {}
    for event in events:
        level = event.level.value
        level_counts[level] = level_counts.get(level, 0) + 1

    # 計算每個攝影機的事件數量
    camera_counts = {}
    for event in events:
        camera_counts[event.camera_id] = camera_counts.get(event.camera_id, 0) + 1

    return {
        "total_events": len(events),
        "type_counts": type_counts,
        "level_counts": level_counts,
        "camera_counts": camera_counts,
        "time_range": {
            "start": min(e.timestamp for e in events) if events else None,
            "end": max(e.timestamp for e in events) if events else None
        }
    }