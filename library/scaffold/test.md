# 路徑
要先進入library目錄


# 切換模型

修改.env檔中的OPENROUTER_MODEL

可用的模型如

google/gemini-2.5-pro
openai/gpt-5-chat
anthropic/claude-sonnet-4

openai/gpt-oss-120b
openai/gpt-oss-20b
openai/gpt-oss-20b:free


## sensemaker_scaffold001.ts測試


### 先準備好/files/comments.csv


### 在根目錄，執行

```
npx ts-node ./library/runner-cli/runner_openrouter.ts \
  --outputBasename out \
  --inputFile "./files/comments.csv" \
  --additionalContext "Description of the conversation"
  ```

### 在根目錄，執行

```bash
npx ts-node ./library/scaffold/sensemaker_scaffold001.ts
```


## JSON 結構化輸出測試

```bash
npx ts-node ./library/scaffold/JSON_ai_prompt.ts "創建一個虛構人物的資料"
```



## Hello World 測試

這是一個簡單的 Hello World 程式。

```bash
npx ts-node ./library/scaffold/hello_world.ts
```

預期輸出

```
Hello world
```

---

## AI Prompt 測試程式

這是一個可以串接 OpenRouter API 的測試程式。

### 使用方法

```bash
npx ts-node ./library/scaffold/simple_ai_prompt.ts "你的問題"
```

### 範例

```bash
# 簡單測試
npx ts-node ./library/scaffold/simple_ai_prompt.ts "測試"

# 中文問題
npx ts-node ./library/scaffold/simple_ai_prompt.ts "請用繁體中文回答：什麼是人工智慧？"
```

### 預期輸出

程式會顯示：
- 使用的模型名稱
- API 端點
- 你的問題
- AI 的回應

### 注意事項

- 需要先在 `library/.env` 檔案中設定 `OPENROUTER_API_KEY` 和 `OPENROUTER_MODEL`
- 確保網路連線正常
- 免費模型可能有使用限制

