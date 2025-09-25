# Overview 多語言功能測試說明

## 概述

這個測試文件專門測試 `overview.ts` 的多語言功能，包括：
- 多語言 prompt 生成
- 真實 LLM 調用測試
- 語言檢測和驗證
- 錯誤處理和邊界情況

## 前置要求

### 1. 環境變數設定

在運行測試之前，需要設定 OpenRouter API Key：

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

### 2. 依賴安裝

確保已安裝所有必要的依賴：

```bash
cd library
npm install
```

## 運行測試

### 完整測試套件

```bash
cd library
npx ts-node scaffold/overview_multilang_test.ts
```

### 單個測試函數

```bash
# 測試多語言 prompt 生成
npx ts-node -e "
import { testMultilangPromptGeneration } from './scaffold/overview_multilang_test';
testMultilangPromptGeneration();
"

# 測試 LLM 語言檢測
npx ts-node -e "
import { testLLMLanguageDetection } from './scaffold/overview_multilang_test';
testLLMLanguageDetection();
"
```

## 測試內容

### 1. 多語言 Prompt 生成測試

- 測試 6 種語言：英文、繁體中文、簡體中文、法文、西班牙文、日文
- 驗證 prompt 是否包含對應語言的內容
- 檢查主題名稱是否正確插入

### 2. OverviewSummary 類測試

- 使用真實的 OpenRouter 模型
- 測試不同語言下的摘要生成
- 驗證回應格式和內容

### 3. LLM 語言檢測測試

- 調用真實 LLM 生成回應
- 檢測回應是否使用正確的語言
- 計算語言匹配率

### 4. 錯誤處理測試

- 測試無效語言處理
- 測試空主題列表處理
- 測試邊界情況

### 5. Prompt 一致性測試

- 檢查所有語言版本的 prompt 結構
- 驗證必要元素的存在
- 確保多語言版本的一致性

## 預期輸出

### 成功情況

```
🚀 開始執行 Overview.ts 多語言功能測試套件...

🧪 開始測試多語言 prompt 生成...

📝 測試 one-shot prompt 生成:
✅ en: prompt 生成成功 (長度: 1234 字符)
   ✓ en: 包含英文內容
   ✓ en: 包含主題名稱
✅ zh-TW: prompt 生成成功 (長度: 1456 字符)
   ✓ zh-TW: 包含中文內容
   ✓ zh-TW: 包含主題名稱
...

🧪 開始測試 OverviewSummary 類的多語言功能（使用真實 LLM）...

🌐 測試語言: en
   📝 測試 oneShotSummary...
   ✅ oneShotSummary 完成，耗時: 2345ms
   📊 結果長度: 567 字符
   📝 結果預覽: * **Topic A (30%):** This is a summary...
   ✓ en: 結果包含英文內容
   ✓ en: 結果格式正確（包含 markdown 列表）
...
```

### 錯誤情況

如果沒有設定 API Key：

```
⚠️ 未設定 OPENROUTER_API_KEY 環境變數，跳過 LLM 測試
💡 請設定環境變數：請先 .env 中設定OPENROUTER_API_KEY 環境變數
```

## 故障排除

### 1. API Key 錯誤

```
❌ Error: OpenRouter API error: 401 Unauthorized
```

**解決方案：**
- 檢查 API Key 是否正確
- 確認 OpenRouter 帳戶是否有效
- 檢查 API 配額是否充足

### 2. 網絡錯誤

```
❌ Error: fetch failed
```

**解決方案：**
- 檢查網絡連接
- 確認防火牆設定
- 嘗試使用 VPN

### 3. 類型錯誤

```
❌ Type 'MockModel' is not assignable to type 'Model'
```

**解決方案：**
- 確保使用正確的 Model 實現
- 檢查導入路徑
- 驗證類型定義

## 自定義測試

### 添加新語言

1. 在 `testLanguages` 數組中添加新語言代碼
2. 在 `languageIndicators` 中添加對應的關鍵詞
3. 運行測試驗證新語言支援

### 修改測試數據

```typescript
// 修改測試主題
const testTopicNames = ["新主題 A (40%)", "新主題 B (60%)"];

// 修改測試內容
const topicsSummary: SummaryContent = {
  title: "自定義測試摘要",
  text: "這是自定義的測試內容",
  citations: [],
  subContents: []
};
```

## 性能考量

- **API 調用次數**：每個語言測試會調用一次 LLM
- **響應時間**：取決於 OpenRouter 的響應速度
- **成本**：每次 API 調用都會產生費用

## 注意事項

1. **API 限制**：注意 OpenRouter 的速率限制
2. **成本控制**：測試會消耗 API 配額
3. **環境隔離**：建議在測試環境中運行
4. **數據隱私**：測試數據不應包含敏感信息

## 貢獻

如果發現問題或需要改進，請：
1. 檢查錯誤日誌
2. 驗證環境設定
3. 提交 issue 或 pull request
