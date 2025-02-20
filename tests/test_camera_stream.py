import cv2
import requests
import numpy as np

def test_camera_stream():
    """測試攝影機串流功能"""
    # 連接到串流端點
    stream_url = "http://localhost:8000/api/v1/camera/stream/0"
    response = requests.get(stream_url, stream=True)

    if response.status_code == 200:
        print("成功連接到串流端點")

        # 讀取串流內容
        bytes_data = bytes()
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]

                # 解碼影像
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                # 顯示影像
                cv2.imshow('Camera Stream Test', frame)

                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_stream()