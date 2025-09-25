# AI Prompt 測試程式

這個程式可以讓你測試 OpenRouter API 的串接，使用指定的模型來回應你的問題。

## 前置需求

1. 在 `.env` 檔案中設定以下環境變數：
   ```bash
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   OPENROUTER_MODEL=openai/gpt-oss-120b
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   ```

2. 確保已安裝所需套件：
   ```bash
   npm install
   ```

## 使用方法

### 基本用法
```bash
npx ts-node scaffold/simple_ai_prompt.ts "你的問題"
```

### 範例
```bash
# 簡單問題
npx ts-node scaffold/simple_ai_prompt.ts "你好，請介紹一下你自己"

# 複雜問題
npx ts-node scaffold/simple_ai_prompt.ts "請解釋什麼是機器學習，並給出三個實際應用例子"
```

## 程式功能

- 接受命令列參數作為 prompt
- 自動載入 `.env` 檔案中的環境變數
- 使用 OpenRouter API 發送請求
- 顯示 AI 模型的回應
- 錯誤處理和參數驗證

## 環境變數說明

- `OPENROUTER_API_KEY`: 你的 OpenRouter API 金鑰
- `OPENROUTER_MODEL`: 要使用的模型名稱（例如：openai/gpt-4o-mini）
- `OPENROUTER_BASE_URL`: OpenRouter API 的基礎 URL（可選，預設為 https://openrouter.ai/api/v1）

## 注意事項

- 確保你的 OpenRouter API 金鑰有效且有足夠的額度
- 模型名稱必須是 OpenRouter 支援的模型
- 程式會自動處理 API 錯誤並顯示詳細的錯誤訊息
