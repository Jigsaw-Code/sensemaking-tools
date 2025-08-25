# OpenRouter 整合指南

本指南說明如何在 Sensemaker 專案中使用 OpenRouter 模型。

## 環境設定

### 方式 1：系統環境變數（推薦用於生產環境）

設定系統環境變數：

```bash
export OPENROUTER_API_KEY="your-api-key"
export OPENROUTER_MODEL="openai/gpt-4"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export DEFAULT_OPENROUTER_PARALLELISM="5"
```

### 方式 2：.env 檔案（僅用於開發環境）

在 `library` 目錄下創建 `.env` 檔案：

```bash
OPENROUTER_API_KEY=your-api-key
OPENROUTER_MODEL=openai/gpt-4
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_OPENROUTER_PARALLELISM=5
```

**注意**：`.env` 檔案只在 `NODE_ENV !== 'production'` 時才會被載入，確保生產環境的安全性。

### 環境變數優先順序

1. **系統環境變數**（最高優先級）
2. **.env 檔案**（僅開發環境）
3. **預設值**（最低優先級）

## 使用方式

### 基本使用

```typescript
import { createOpenRouterModelFromEnv } from './src/models/openrouter_model';

// 自動從環境變數建立模型
const model = createOpenRouterModelFromEnv();
```

### 直接建立

```typescript
import { OpenRouterModel } from './src/models/openrouter_model';

const model = new OpenRouterModel(
  'your-api-key',
  'openai/gpt-4',
  'https://openrouter.ai/api/v1'
);
```

## 部署說明

### Docker 環境

```dockerfile
ENV OPENROUTER_API_KEY=your-api-key
ENV OPENROUTER_MODEL=openai/gpt-4
ENV OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
ENV DEFAULT_OPENROUTER_PARALLELISM=5
```

### Kubernetes 環境

```yaml
env:
- name: OPENROUTER_API_KEY
  valueFrom:
    secretKeyRef:
      name: openrouter-secret
      key: api-key
- name: OPENROUTER_MODEL
  value: "openai/gpt-4"
- name: OPENROUTER_BASE_URL
  value: "https://openrouter.ai/api/v1"
```

### Serverless 環境

```javascript
// AWS Lambda, Vercel, Netlify 等
process.env.OPENROUTER_API_KEY = 'your-api-key';
process.env.OPENROUTER_MODEL = 'openai/gpt-4';
```

## 瀏覽器環境支援

本套件已設計為瀏覽器友好：

- 優先讀取系統環境變數
- 不依賴 Node.js 特定的檔案系統操作
- 支援 Web Workers 和 Serverless 環境

## 故障排除

### 常見問題

1. **API 金鑰未設定**
   - 檢查 `OPENROUTER_API_KEY` 環境變數
   - 確認 `.env` 檔案格式正確

2. **模型名稱錯誤**
   - 使用正確的 OpenRouter 模型名稱格式
   - 例如：`openai/gpt-4`, `anthropic/claude-3-sonnet`

3. **並發限制問題**
   - 調整 `DEFAULT_OPENROUTER_PARALLELISM` 值
   - 根據你的 OpenRouter 計劃調整

### 除錯模式

設定 `DEBUG_MODE=true` 來啟用詳細日誌：

```bash
export DEBUG_MODE=true
```
