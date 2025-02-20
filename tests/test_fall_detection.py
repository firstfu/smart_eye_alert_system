import asyncio
import websockets
import json
import cv2
import numpy as np

async def test_fall_detection():
    """測試跌倒偵測功能"""
    # 連接 WebSocket
    uri = "ws://localhost:8000/api/v1/camera/ws/0"

    async with websockets.connect(uri) as websocket:
        print("已連接到 WebSocket")

        try:
            while True:
                # 接收偵測結果
                response = await websocket.recv()
                data = json.loads(response)

                # 顯示偵測結果
                print("\n檢測結果:")
                print(f"是否跌倒: {data['is_fall']}")
                print(f"置信度: {data['confidence']:.2f}")
                print(f"身體角度: {data['details']['angle']:.2f}")
                print(f"移動速度: {data['details']['movement_speed']:.2f}")

                # 如果偵測到跌倒
                if data['is_fall'] and data['confidence'] > 0.7:
                    print("\n⚠️ 警告：偵測到跌倒！")

        except KeyboardInterrupt:
            print("\n測試結束")

if __name__ == "__main__":
    asyncio.run(test_fall_detection())