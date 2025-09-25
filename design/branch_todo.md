# Sensemaker 專案：從 Vertex AI 遷移到 OpenRouter 待辦清單

## 工程基本方向

1. 我想做的改良，是以open router來代替Vertex AI
2. .env檔在library目錄下，使用者變更.env中想用的模型參數，即可控制本程式使用的模型
3. 創建新的檔案而非直接變更舊的檔案，以便舊檔案和原專案同步。

## 第零階段：標記並複製核心檔案

### 0.1 複製核心類型定義
- [x] 分析 `library/src/types.ts` → 分析並理解所有 TypeBox Schema 定義 (無 Vertex AI 依賴)
- [x] 分析 `library/src/models/model.ts` → 理解抽象 Model 類別介面 (無 Vertex AI 依賴)
- [x] 複製 `library/src/models/vertex_model.ts` 至 `library/src/models/openrouter_model.ts` → 分析 Vertex AI 實作細節 (有 Vertex AI 依賴)

### 0.2 分析 Prompt 處理函數
- [x] 分析 `library/src/sensemaker_utils.ts` → 分析 `getPrompt` 和 `getAbstractPrompt` 函數 (無 Vertex AI 依賴，只有 model_util 依賴)
- [x] 分析 `library/src/tasks/summarization_subtasks/` 目錄 → 分析所有 prompt 指令範例 (無 Vertex AI 依賴)

### 0.3 複製測試檔案作為參考
- [x] 複製 `library/src/models/vertex_model.test.ts` 至 `library/src/models/openrouter_model.test.ts`  → 理解測試模式和驗證邏輯 (有 Vertex AI 依賴)
- [x] 分析 `library/src/types.test.ts` → 理解類型驗證測試 (無 Vertex AI 依賴)

### 0.4 複製工具和配置檔案
- [x] 分析 `library/src/models/model_util.ts` → 分析常數和工具函數 (無 Vertex AI 依賴，只是常數定義)
- [x] 分析 `library/package.json` → 分析依賴關係和腳本 (無 Vertex AI 依賴)

### 0.5 分析現有 Schema 結構
- [x] 分析 `FlatTopic`, `NestedTopic`, `Topic` 等核心類型
- [x] 分析 `Comment`, `VoteTally`, `Summary` 等資料結構
- [x] 分析 `checkDataSchema` 函數的實作邏輯
- [x] 分析所有 prompt 指令的格式和內容

## 第一階段：基礎架構建立

### 1.1 建立 OpenRouter 模型類別
- [x] 建立 `library/src/models/openrouter_model.ts` 檔案
- [x] 實作 `Model` 抽象類別的具體實作
- [x] 整合 OpenRouter API 呼叫
- [x] 實作速率限制和重試機制
- [x] 支援結構化輸出 (JSON Schema)

### 1.2 建立 OpenRouter 工具類別
- [ ] 建立 `library/src/models/openrouter_util.ts` 檔案
- [x] 定義 OpenRouter 相關常數
- [x] 實作 API 金鑰管理
- [x] 定義預設模型設定
- [ ] 實作請求格式轉換

### 1.3 更新環境變數設定
- [x] 建立 `.env.example` 檔案
- [x] 新增 `dotenv` 依賴管理環境變數
- [x] 定義 OpenRouter API 金鑰
- [ ] 設定預設模型選擇
- [ ] 設定速率限制參數

## 第二階段：核心功能實作

### 2.1 實作 OpenRouter API 整合
- [ ] 實作 HTTP 請求到 OpenRouter API
- [ ] 處理不同的模型回應格式
- [ ] 實作錯誤處理和重試邏輯
- [ ] 支援串流和非串流回應

### 2.2 實作 Prompt 處理
- [ ] 保持與現有 `getPrompt` 函數的相容性
- [ ] 實作 OpenRouter 特定的 prompt 格式化
- [ ] 支援不同模型的 prompt 要求

### 2.3 實作結構化輸出
- [ ] 實作 JSON Schema 驗證
- [ ] 處理不同模型的 JSON 輸出格式
- [ ] 實作回應解析和錯誤處理

## 第三階段：測試和驗證

### 3.1 單元測試
- [ ] 建立 `library/src/models/openrouter_model.test.ts` 檔案
- [ ] 測試模型初始化和設定
- [ ] 測試 prompt 處理和 API 呼叫
- [ ] 測試錯誤處理和重試機制
- [ ] 測試結構化輸出驗證

### 3.2 整合測試
- [ ] 建立 `library/src/models/openrouter_integration.test.ts` 檔案
- [ ] 測試與現有 prompt 函數的整合
- [ ] 測試與不同任務模組的相容性
- [ ] 測試端到端的工作流程

### 3.3 效能測試
- [ ] 測試 API 回應時間
- [ ] 測試並發處理能力
- [ ] 測試速率限制效果
- [ ] 與 Vertex AI 的效能比較

## 第四階段：配置和部署

### 4.1 模型設定管理
- [ ] 建立 `library/src/models/model_factory.ts` 檔案
- [ ] 實作模型選擇邏輯
- [ ] 支援環境變數配置
- [ ] 實作模型切換機制

### 4.2 配置文件更新
- [ ] 建立 `library/config/models.json` 檔案
- [ ] 定義支援的模型清單
- [ ] 設定模型特定參數
- [ ] 配置預設模型選擇

### 4.3 文檔更新
- [ ] 建立 `library/docs/openrouter_migration.md` 檔案
- [ ] 撰寫遷移指南
- [ ] 更新 API 文檔
- [ ] 提供配置範例

## 第五階段：優化和清理

### 5.1 效能優化
- [ ] 優化 API 呼叫效率
- [ ] 改善錯誤處理邏輯
- [ ] 優化記憶體使用

### 5.2 程式碼清理
- [ ] 移除未使用的程式碼
- [ ] 改善程式碼註解
- [ ] 統一程式碼風格

### 5.3 最終測試
- [ ] 執行完整的測試套件
- [ ] 驗證所有功能正常運作
- [ ] 檢查效能指標

## 成功標準檢查清單

### 功能完整性
- [ ] 所有現有測試通過
- [ ] 支援所有現有功能
- [ ] 與現有程式碼完全相容

### 效能指標
- [ ] API 回應時間 < 10 秒 (95th percentile)
- [ ] 錯誤率 < 1%
- [ ] 並發處理能力 >= 現有實作

### 品質標準
- [ ] 程式碼覆蓋率 >= 90%
- [ ] 所有 lint 檢查通過
- [ ] 文檔完整且準確

## 後續改進項目

### 短期改進 (1-2 個月)
- [ ] 支援更多 OpenRouter 模型
- [ ] 實作模型效能比較工具
- [ ] 改善錯誤報告和監控

### 長期改進 (3-6 個月)
- [ ] 實作模型自動選擇
- [ ] 支援混合模型策略
- [ ] 實作成本優化建議
