# 智眼警示系統 - 技術棧分析文件

## 1. 系統架構概觀

### 1.1 整體架構

```
[攝影機] -> [影像擷取模組] -> [AI 分析模組] -> [事件處理模組] -> [通知模組]
                ↑                    ↑               ↑              ↑
                └──────────── [核心服務模組] ──────────────────────┘
                                    ↑
                            [前端管理介面]
```

### 1.2 模組職責

- 影像擷取模組：處理攝影機串流
- AI 分析模組：進行跌倒偵測
- 事件處理模組：判斷警報觸發
- 通知模組：發送警報
- 核心服務模組：協調各模組運作
- 前端管理介面：系統配置與監控

## 2. 前端技術棧

### 2.1 核心框架

- Next.js 14 (App Router)
  - 使用原因：
    - 開發效率高
    - 內建 API Routes
    - 檔案系統路由
    - 效能優化工具

### 2.2 主要依賴

```json
{
  "dependencies": {
    "next": "14.x",
    "react": "18.x",
    "typescript": "5.x",
    "tailwindcss": "3.x",
    "socket.io-client": "4.x",
    "zustand": "4.x"
  }
}
```

### 2.3 前端架構

```
src/
├── app/                # App Router 結構
│   ├── layout.tsx     # 根布局
│   ├── page.tsx       # 首頁
│   ├── dashboard/     # 儀表板
│   └── settings/      # 設定頁面
├── components/         # 共用元件
├── hooks/             # 自定義 Hooks
└── lib/               # 工具函數
```

## 3. 後端技術棧

### 3.1 核心框架

- FastAPI
  - 使用原因：
    - 高效能非同步處理
    - 自動 API 文檔
    - WebSocket 支援
    - 易於整合 AI 模型

### 3.2 主要依賴

```python
# requirements.txt
fastapi==0.104.x
uvicorn==0.24.x
python-multipart==0.0.6
opencv-python==4.8.x
mediapipe==0.10.x
ultralytics==8.x.x
python-socketio==5.x.x
```

### 3.3 後端架構

```
backend/
├── app/
│   ├── api/          # API 端點
│   ├── core/         # 核心邏輯
│   ├── models/       # 資料模型
│   └── services/     # 業務邏輯
├── ml/
│   ├── detectors/    # 偵測器
│   └── processors/   # 影像處理
└── utils/            # 工具函數
```

## 4. AI 模型架構

### 4.1 多階段分析流程

1. 影像預處理

   ```python
   def preprocess_frame(frame):
       # 調整大小為 640x480
       # 正規化像素值
       # 去噪處理
   ```

2. 初步偵測（即時）

   ```python
   def quick_detection(frame):
       # 使用 MediaPipe 進行姿態估計
       # 基於規則的快速判斷
       # 返回初步結果
   ```

3. LLM 分析（非同步）

   ```python
   async def llm_analysis(detection_result, context):
       # 將偵測結果轉換為文字描述
       description = convert_to_text(detection_result)

       # LLM 分析提示詞
       prompt = f"""
       基於以下場景描述，判斷是否為緊急情況：
       場景：{description}
       上下文：{context}

       請分析：
       1. 風險等級（高/中/低）
       2. 建議採取的行動
       3. 可能的誤判因素
       """

       # 調用 LLM API
       response = await call_llm_api(prompt)
       return parse_llm_response(response)
   ```

### 4.2 LLM 整合方案

#### 4.2.1 模型選擇

- 主要模型：OpenAI GPT-4

  - 用於深度場景分析
  - 非即時決策支援

- 備用方案：本地 LLM
  - Llama 2 (7B-Chat)
  - GGUF 量化版本
  - 用於離線場景

#### 4.2.2 優化策略

- 使用 API 快取
- 批次處理請求
- 結果本地儲存

#### 4.2.3 成本控制

- 僅在高風險事件觸發 LLM
- 使用 token 配額管理
- 定期評估使用效益

### 4.3 混合分析流程

```python
class HybridDetectionSystem:
    def __init__(self):
        self.quick_detector = MediaPipeDetector()
        self.llm_analyzer = LLMAnalyzer()
        self.event_cache = Cache()

    async def process_frame(self, frame):
        # 1. 快速偵測
        quick_result = self.quick_detector.detect(frame)

        # 2. 風險評估
        if quick_result.risk_level >= THRESHOLD:
            # 3. LLM 深度分析
            context = self.event_cache.get_recent_events()
            llm_result = await self.llm_analyzer.analyze(
                quick_result, context
            )

            # 4. 決策整合
            final_decision = self.integrate_results(
                quick_result, llm_result
            )

            return final_decision

        return quick_result
```

### 4.4 效能考量

#### 4.4.1 即時性保證

- 主要偵測仍依賴 MediaPipe
- LLM 分析在背景非同步進行
- 使用快取減少 API 調用

#### 4.4.2 資源使用

- CPU: 主要用於影像處理
- GPU: 可選用於本地 LLM（如有）
- 記憶體: < 4GB（含本地 LLM）

#### 4.4.3 成本估算

- OpenAI API: 每千次分析約 US$0.1
- 本地部署: 僅需硬體資源
- 混合模式: 依使用情況彈性調整

## 5. 部署架構

### 5.1 Docker 配置

```yaml
# docker-compose.yml
version: "3.8"
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
```

### 5.2 本地部署流程

1. 環境檢查
2. Docker 安裝
3. 設定檔配置
4. 啟動服務
5. 攝影機設定

## 6. 效能優化

### 6.1 影像處理優化

- 降低解析度：640x480
- 跳幀處理：每秒 10 幀
- 區域檢測：只處理移動區域

### 6.2 系統效能指標

- CPU 使用率 < 50%
- 記憶體使用 < 2GB
- 延遲時間 < 500ms

## 7. 安全性考量

### 7.1 基礎安全措施

- HTTPS 加密
- JWT 認證
- 影像資料加密儲存

### 7.2 隱私保護

- 即時影像不儲存
- 事件記錄最多保留 7 天
- 影像模糊化處理

## 8. 擴展性設計

### 8.1 模組化介面

```python
class DetectorBase:
    async def detect(self, frame):
        pass

class NotifierBase:
    async def notify(self, event):
        pass
```

### 8.2 插件系統

- 自定義偵測器
- 自定義通知方式
- 自定義警報規則

## 9. 開發流程

### 9.1 第一週目標

- 基礎架構搭建
- 攝影機串流測試
- MediaPipe 整合

### 9.2 第二週目標

- 跌倒偵測邏輯
- 通知系統整合
- 前端雛形

### 9.3 第三週目標

- 系統整合測試
- 效能優化
- 部署流程建立

## 10. 技術風險與對策

### 10.1 效能風險

- 問題：系統負載過高
- 對策：
  1. 使用輕量化模型
  2. 實作快取機制
  3. 最佳化影像處理

### 10.2 準確度風險

- 問題：誤報/漏報
- 對策：
  1. 多重確認機制
  2. 自適應閾值
  3. 人工確認選項
