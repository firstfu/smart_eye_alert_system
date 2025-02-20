import requests
from datetime import datetime

def test_notification():
    """測試通知功能"""
    # 模擬跌倒事件資料
    event_data = {
        "event_type": "fall_detection",
        "risk_level": "high",
        "confidence": 0.95,
        "details": {
            "angle": 75.5,
            "movement_speed": 0.8,
            "message": "測試通知：偵測到跌倒事件"
        },
        "camera_id": "test_camera",
        "location": "測試位置",
        "timestamp": datetime.now().timestamp()
    }

    # 發送事件到後端
    response = requests.post(
        "http://localhost:8000/api/v1/camera/notify",
        json=event_data
    )

    if response.status_code == 200:
        print("通知發送成功")
        print(response.json())
    else:
        print(f"通知發送失敗: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_notification()