import cv2
import time
from typing import Optional

def test_camera_connection(camera_source: str, name: str = "Camera Test") -> bool:
    """測試攝影機連接

    Args:
        camera_source: 可以是數字（本地攝影機）或 RTSP URL（網路攝影機）
        name: 視窗標題

    Returns:
        bool: 連接是否成功
    """
    try:
        # 如果是數字字串，轉換為整數
        if camera_source.isdigit():
            camera_source = int(camera_source)

        # 開啟攝影機
        cap = cv2.VideoCapture(camera_source)

        if not cap.isOpened():
            print(f"無法開啟攝影機: {camera_source}")
            return False

        # 獲取攝影機資訊
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f"\n攝影機資訊:")
        print(f"解析度: {width}x{height}")
        print(f"FPS: {fps}")

        # 測試讀取影像
        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取影像")
                break

            # 計算實際 FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            actual_fps = frame_count / elapsed_time

            # 在影像上顯示 FPS
            cv2.putText(
                frame,
                f"FPS: {actual_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # 顯示影像
            cv2.imshow(name, frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return True

    except Exception as e:
        print(f"錯誤: {str(e)}")
        return False

def main():
    """主要測試流程"""
    while True:
        print("\n攝影機測試工具")
        print("1. 測試本地攝影機")
        print("2. 測試網路攝影機")
        print("3. 退出")

        choice = input("請選擇 (1-3): ")

        if choice == "1":
            camera_id = input("請輸入攝影機編號 (預設 0): ") or "0"
            test_camera_connection(camera_id)

        elif choice == "2":
            ip = input("請輸入攝影機 IP: ")
            port = input("請輸入埠號 (預設 554): ") or "554"
            username = input("請輸入使用者名稱: ")
            password = input("請輸入密碼: ")
            stream_path = input("請輸入串流路徑 (例如 stream1): ")

            rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/{stream_path}"
            test_camera_connection(rtsp_url)

        elif choice == "3":
            break

        else:
            print("無效的選擇，請重試")

if __name__ == "__main__":
    main()