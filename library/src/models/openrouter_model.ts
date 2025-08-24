// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Module to interact with models available through OpenRouter, including various
// AI models from different providers like OpenAI, Anthropic, Google, etc.

import { OpenAI } from "openai";
import { Model } from "./model";
import { TSchema, Static } from "@sinclair/typebox";

// Import localization system
import { getLanguagePrefix, type SupportedLanguage } from "../../templates/l10n";

/**
 * 獨立的 buffer 上下文，用於避免並行處理時的 buffer 干擾
 */
interface StreamBufferContext {
  buffer: string;
  chunks: string[];
  chunkCount: number;
  startTime: number;
  timeoutMs: number;
}

export class OpenRouterModel extends Model {
  private openai: OpenAI;
  private modelName: string;

  constructor(apiKey: string, modelName: string = "openai/gpt-oss-120b") {
    // 設定更小的批次大小，適合處理大筆資料
    super(50);
    this.modelName = modelName;
    this.openai = new OpenAI({
      apiKey: apiKey,
      baseURL: "https://openrouter.ai/api/v1",
      defaultHeaders: {
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "Your App Name",
      },
    });
  }

  async generateText(prompt: string, output_lang: SupportedLanguage = "en"): Promise<string> {
    return await this.callLLM(prompt, () => true, undefined, output_lang);
  }

  async generateData(prompt: string, schema: TSchema, output_lang: SupportedLanguage = "en"): Promise<Static<typeof schema>> {
    try {
      const response = await this.callLLM(prompt, validateResponse, schema, output_lang);
      if (!response) {
        throw new Error("Empty response from OpenRouter API");
      }
      const parsed = JSON.parse(response);
      
      // 處理 LLM 可能回傳的包裝格式，如 {"items": [...]}
      let processedData = parsed;
      console.log(`   🔍 Raw parsed response:`, JSON.stringify(parsed, null, 2));
      
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        // 檢查是否有常見的包裝鍵
        const wrapperKeys = ['items', 'data', 'result', 'content', 'output'];
        for (const key of wrapperKeys) {
          if (key in parsed && Array.isArray(parsed[key])) {
            console.log(`   🔧 Detected wrapped array in '${key}' key, extracting...`);
            processedData = parsed[key];
            break;
          }
        }
      }
      
      console.log(`   🔍 Final processed data:`, JSON.stringify(processedData, null, 2));
      
      // 在 Cloudflare Workers 環境中，避免使用 TypeBox 編譯器
      // 改用簡單的 JSON 驗證
      if (schema && Array.isArray(schema)) {
        // 如果 schema 是數組類型，確保回應也是數組
        if (!Array.isArray(processedData)) {
          console.error('Schema expects array but response is not array:', typeof processedData, processedData);
          throw new Error('Response format error: expected array but got ' + typeof processedData);
        }
      }
      
      // 基本類型檢查（避免使用 TypeBox 編譯器）
      if (schema && typeof schema === 'object' && 'type' in schema) {
        if (schema.type === 'array' && !Array.isArray(processedData)) {
          throw new Error('Response format error: expected array but got ' + typeof processedData);
        }
        if (schema.type === 'object' && (typeof processedData !== 'object' || Array.isArray(processedData))) {
          throw new Error('Response format error: expected object but got ' + typeof processedData);
        }
      }
      
      return processedData;
    } catch (error) {
      console.error('Error in generateData:', error);
      throw error;
    }
  }

  async callLLM(prompt: string, validator: (response: string) => boolean = () => true, schema?: TSchema, output_lang: SupportedLanguage = "en", maxRetries: number = 5): Promise<string> {
    // Get language prefix from localization system
    const languagePrefix = getLanguagePrefix(output_lang);
    
    const requestBody = {
      model: this.modelName,
      messages: [
        { role: "system" as const, content: languagePrefix },
        { role: "user" as const, content: prompt }
      ],
      max_tokens: 16000,
      temperature: 0,
      stream: true,
      n: 1,
      stop: null,
      presence_penalty: 0,
      frequency_penalty: 0,
    };
    
          // 如果有 schema，設定結構化輸出
      if (schema) {
        // OpenRouter 支援 json_schema 格式，格式與官方文檔一致
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (requestBody as any).response_format = {
          type: "json_schema",
          json_schema: {
            name: "response",
            strict: true,
            schema: schema
          }
        };
      }

    let lastError: Error | null = null;
    
    // 重試邏輯
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`   🔄 Attempt ${attempt}/${maxRetries}`);
        
        // 使用 fetch API 發送 streaming 請求
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${this.openai.apiKey}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/your-repo',
            'X-Title': 'Your App Name',
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`OpenRouter API error: ${response.status} ${response.statusText}`);
        }

        // 處理 streaming 回應
        const streamedResponse = await this.processStreamingResponse(response);
        
        // 在驗證前記錄詳細資訊
        console.log('🔍 Streaming Response Debug Info:');
        console.log('   Original Response Length:', streamedResponse.length);
        console.log('   Response Preview (first 200 chars):', streamedResponse.substring(0, 200));
        console.log('   Response Preview (last 200 chars):', streamedResponse.substring(Math.max(0, streamedResponse.length - 200)));
        
        // 在最後整併完成後進行驗證
        if (!validator(streamedResponse)) {
          const error = new Error(`Response validation failed on attempt ${attempt}/${maxRetries}`);
          console.error(`❌ Response validation failed on attempt ${attempt}!`);
          console.error('   Validator function:', validator.toString());
          console.error('   Response that failed validation (preview):', streamedResponse.substring(0, 500) + '...');
          
          if (attempt === maxRetries) {
            console.error('   Full Response:', streamedResponse);
            throw error;
          }
          
          lastError = error;
          console.log(`   ⏳ Retrying in ${attempt * 1000}ms...`);
          await this.sleep(attempt * 1000); // 指數退避：1s, 2s, 3s
          continue;
        }
        
        console.log(`✅ Response validation passed successfully on attempt ${attempt}`);
        return streamedResponse;
        
      } catch (error) {
        lastError = error as Error;
        console.error(`❌ Error in callLLM attempt ${attempt}:`, error);
        
        if (attempt === maxRetries) {
          console.error(`💀 All ${maxRetries} attempts failed. Throwing last error.`);
          throw lastError;
        }
        
        console.log(`   ⏳ Retrying in ${attempt * 1000}ms...`);
        await this.sleep(attempt * 1000); // 指數退避
      }
    }
    
    throw lastError || new Error("Unexpected error in retry logic");
  }

  /**
   * 睡眠指定的毫秒數
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * 創建新的 buffer 上下文，確保每個請求都有獨立的狀態
   */
  private createBufferContext(): StreamBufferContext {
    return {
      buffer: '',
      chunks: [],
      chunkCount: 0,
      startTime: Date.now(),
      timeoutMs: 300000, // 5 分鐘超時
    };
  }

  /**
   * 處理 streaming 回應 - 支援 OpenRouter 的 SSE 格式
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async processStreamingResponse(stream: any): Promise<string> {
    try {
      // 檢查是否是 Response 物件 (fetch API 回應)
      if (stream && stream.body && typeof stream.body.getReader === 'function') {
        return await this.handleFetchResponse(stream);
      }
      
      // 檢查是否是 AsyncIterable (OpenAI SDK streaming 格式)
      if (stream && typeof stream[Symbol.asyncIterator] === 'function') {
        return await this.handleAsyncIterable(stream);
      }
      
      // 檢查是否是陣列格式的 chunks
      if (Array.isArray(stream)) {
        return this.handleChunkArray(stream);
      }
      
      // 檢查是否是單一回應物件
      if (stream && stream.choices && stream.choices[0]) {
        const choice = stream.choices[0];
        if (choice.message && choice.message.content) {
          return this.processStreamedResponse(choice.message.content);
        }
        if (choice.delta && choice.delta.content) {
          return this.processStreamedResponse(choice.delta.content);
        }
      }
      
      // 如果都無法處理，嘗試直接提取內容
      const response = JSON.stringify(stream);
      console.warn("Unable to parse streaming response, using raw content:", response);
      return this.processStreamedResponse(response);
      
    } catch (error) {
      console.error('Error processing streaming response:', error);
      throw new Error('Failed to process streaming response');
    }
  }

  /**
   * 處理 AsyncIterable (OpenAI SDK streaming 格式)
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleAsyncIterable(stream: AsyncIterable<any>): Promise<string> {
    const context = this.createBufferContext();
    
    try {
      for await (const chunk of stream) {
        if (chunk && chunk.choices && chunk.choices[0]) {
          const choice = chunk.choices[0];
          if (choice.delta && choice.delta.content) {
            context.chunks.push(choice.delta.content);
          }
        }
      }
    } catch (error) {
      console.error('Error processing async iterable stream:', error);
      throw error;
    }
    
    const fullResponse = context.chunks.join('');
    return this.processStreamedResponse(fullResponse);
  }

  /**
   * 處理 fetch API 回應 (OpenRouter 官網推薦方式)
   */
  private async handleFetchResponse(response: Response): Promise<string> {
    console.log('📡 Starting fetch API streaming response processing...');
    
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    // 為每個請求創建獨立的 buffer 上下文
    const context = this.createBufferContext();
    const decoder = new TextDecoder();

    try {
      while (true) {
        // 檢查超時
        if (Date.now() - context.startTime > context.timeoutMs) {
          console.log(`   ⏰ Timeout after ${context.timeoutMs}ms, forcing stream completion`);
          break;
        }
        
        const { done, value } = await reader.read();
        if (done) {
          console.log('   Stream completed, total chunks received:', context.chunkCount);
          break;
        }

        // Append new chunk to buffer
        const decodedChunk = decoder.decode(value, { stream: true });
        context.buffer += decodedChunk;
        context.chunkCount++;
        
        // 限制 chunk 數量，防止無限循環
        if (context.chunkCount > 100000) {
          console.log('   ⚠️ Reached maximum chunk limit (100000), forcing completion');
          break;
        }
        
        // console.log(`   Received chunk ${context.chunkCount}, buffer size: ${context.buffer.length}`);

        // Process complete lines from buffer
        let doneSignalReceived = false;
        while (true) {
          const lineEnd = context.buffer.indexOf('\n');
          if (lineEnd === -1) break;

          const line = context.buffer.slice(0, lineEnd).trim();
          context.buffer = context.buffer.slice(lineEnd + 1);

          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              console.log('   Received [DONE] signal');
              doneSignalReceived = true;
              // 不要立即返回，繼續處理 buffer 中剩餘的內容
              break;
            }

            try {
              const parsed = JSON.parse(data);
              const content = parsed.choices ? parsed.choices[0]?.delta?.content : parsed.content;
              if (content) {
                context.chunks.push(content);
                // console.log(`   Extracted content chunk: "${content}"`);
              } //else {
                // 顯示實際收到的 JSON 結構，幫助診斷
                //if (context.chunkCount <= 5 || context.chunkCount % 1000 === 0) {
                //  // console.log(`   No content found. JSON structure: ${JSON.stringify(parsed).substring(0, 200)}`);
                //}
              //}
            } catch (e) {
              // Ignore invalid JSON
              console.warn('   Invalid JSON in streaming response:', e, 'Line:', line);
            }
          }
        }
        
        // 如果收到 [DONE] 信號，處理完 buffer 中剩餘的內容後退出主循環
        if (doneSignalReceived) {
          // 處理 buffer 中剩餘的內容
          if (context.buffer.length > 0) {
            console.log(`   Processing remaining buffer content (${context.buffer.length} chars): "${context.buffer}"`);
            // 嘗試解析剩餘的 buffer 內容
            try {
              const parsed = JSON.parse(context.buffer);
              const content = parsed.choices?.[0]?.delta?.content;
              if (content) {
                context.chunks.push(content);
                console.log(`   Extracted final content chunk: "${content}"`);
              }
            } catch {
              console.log('   Remaining buffer is not valid JSON, skipping');
            }
          }
          break;
        }
      }
    } finally {
      reader.cancel();
      console.log('   Reader cancelled');
    }

    const fullResponse = context.chunks.join('');
    console.log('   Total content length:', fullResponse.length);
    console.log('   Number of content chunks:', context.chunkCount);
    
    return this.processStreamedResponse(fullResponse);
  }

  /**
   * 處理陣列格式的 chunks
   */
  private handleChunkArray(chunks: string[]): string {
    const fullResponse = chunks.join('');
    return this.processStreamedResponse(fullResponse);
  }

  /**
   * 處理可能的 streaming 回應，嘗試修復不完整的回應
   */
  private processStreamedResponse(response: string): string {
    console.log('🔧 Processing streamed response...');
    console.log('   Original length:', response.length);
    console.log('   Original content:', response);
    
    let processedResponse = response;

    // 移除 OpenRouter 的處理註釋
    const beforeProcessing = processedResponse;
    processedResponse = processedResponse.replace(/: OPENROUTER PROCESSING/g, '');
    if (beforeProcessing !== processedResponse) {
      console.log('   Removed OPENROUTER PROCESSING comments');
    }
    
    // 移除多餘的空白行
    processedResponse = processedResponse.replace(/\n\s*\n/g, '\n');
    
    // 檢查並修復 JSON 完整性
    processedResponse = this.fixIncompleteJson(processedResponse);
    
    const finalResponse = processedResponse.trim();
    console.log('   Final processed length:', finalResponse.length);
    console.log('   Final processed content:', finalResponse);
    
    return finalResponse;
  }

  /**
   * 修復不完整的 JSON - 更智能的版本
   */
  private fixIncompleteJson(response: string): string {
    let fixedResponse = response;
    
    console.log('   🔧 Starting JSON repair process...');
    
    // 首先嘗試直接解析，如果成功就直接返回
    try {
      JSON.parse(fixedResponse);
      console.log('   ✅ JSON is already valid, no repair needed');
      return fixedResponse;
    } catch {
      console.log('   ❌ JSON validation failed, starting repair...');
    }
    
    // 嘗試從 Markdown 回應中提取 JSON
    const extractedJson = this.extractJsonFromMarkdown(fixedResponse);
    if (extractedJson) {
      console.log('   🔍 Found JSON in Markdown, attempting to extract...');
      try {
        JSON.parse(extractedJson);
        console.log('   ✅ Successfully extracted valid JSON from Markdown');
        return extractedJson;
      } catch {
        console.log('   ⚠️ Extracted JSON is invalid, continuing with other repair methods...');
      }
    }
    
    // 修復常見的 streaming 問題
    fixedResponse = this.fixStreamingIssues(fixedResponse);
    
    // 嘗試解析修復後的內容
    try {
      JSON.parse(fixedResponse);
      console.log('   ✅ JSON validation passed after streaming fixes');
      return fixedResponse;
    } catch {
      console.log('   ❌ Still invalid after streaming fixes, attempting structural repair...');
    }
    
    // 進行結構性修復
    fixedResponse = this.fixStructuralIssues(fixedResponse);
    
    // 最終驗證
    try {
      JSON.parse(fixedResponse);
      console.log('   ✅ JSON validation passed after structural repair');
      return fixedResponse;
    } catch (error) {
      console.log('   ❌ JSON validation failed after all repair attempts');
      console.log('   Final repair attempt failed:', error);
      
      // 最後的嘗試：找到最後一個完整的 JSON 結構
      return this.findLastValidJson(fixedResponse);
    }
  }
  
  /**
   * 從 Markdown 回應中提取 JSON
   */
  private extractJsonFromMarkdown(response: string): string | null {
    console.log('   🔍 Searching for JSON in Markdown response...');
    
    // 方法 1: 尋找 ```json 代碼塊
    const jsonCodeBlockMatch = response.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonCodeBlockMatch) {
      console.log('   ✅ Found JSON code block');
      return jsonCodeBlockMatch[1].trim();
    }
    
    // 方法 2: 尋找 ``` 代碼塊（沒有 json 標籤）
    const codeBlockMatch = response.match(/```\s*([\s\S]*?)\s*```/);
    if (codeBlockMatch) {
      const content = codeBlockMatch[1].trim();
      // 檢查內容是否看起來像 JSON
      if (content.startsWith('{') || content.startsWith('[')) {
        console.log('   ✅ Found JSON-like code block');
        return content;
      }
    }
    
    // 方法 3: 尋找 JSON 對象（從 { 開始到 } 結束）
    const jsonObjectMatch = response.match(/\{[\s\S]*\}/);
    if (jsonObjectMatch) {
      console.log('   ✅ Found JSON object in response');
      return jsonObjectMatch[0];
    }
    
    // 方法 4: 尋找 JSON 陣列（從 [ 開始到 ] 結束）
    const jsonArrayMatch = response.match(/\[[\s\S]*\]/);
    if (jsonArrayMatch) {
      console.log('   ✅ Found JSON array in response');
      return jsonArrayMatch[0];
    }
    
    // 方法 5: 尋找可能的 JSON 結構（更寬鬆的匹配）
    const possibleJsonMatch = response.match(/(\{[\s\S]*\}|\[[\s\S]*\])/);
    if (possibleJsonMatch) {
      console.log('   ⚠️ Found possible JSON structure, attempting to clean...');
      return this.cleanPossibleJson(possibleJsonMatch[0]);
    }
    
    console.log('   ❌ No JSON found in Markdown response');
    return null;
  }
  
  /**
   * 清理可能的 JSON 結構
   */
  private cleanPossibleJson(possibleJson: string): string {
    let cleaned = possibleJson;
    
    // 移除可能的 Markdown 標記
    cleaned = cleaned.replace(/\*\*/g, ''); // 移除粗體標記
    cleaned = cleaned.replace(/\*/g, '');   // 移除斜體標記
    cleaned = cleaned.replace(/`/g, '');    // 移除代碼標記
    cleaned = cleaned.replace(/#{1,6}\s*/g, ''); // 移除標題標記
    cleaned = cleaned.replace(/-\s*/g, ''); // 移除列表標記
    
    // 移除多餘的空白行
    cleaned = cleaned.replace(/\n\s*\n/g, '\n');
    
    // 嘗試找到 JSON 的開始和結束
    let startIndex = -1;
    let endIndex = -1;
    
    // 尋找 JSON 開始
    if (cleaned.includes('{')) {
      startIndex = cleaned.indexOf('{');
    } else if (cleaned.includes('[')) {
      startIndex = cleaned.indexOf('[');
    }
    
    // 尋找 JSON 結束
    if (cleaned.includes('}')) {
      endIndex = cleaned.lastIndexOf('}') + 1;
    } else if (cleaned.includes(']')) {
      endIndex = cleaned.lastIndexOf(']') + 1;
    }
    
    if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
      cleaned = cleaned.substring(startIndex, endIndex);
      console.log('   🔧 Cleaned JSON structure');
    }
    
    return cleaned;
  }
  
  /**
   * 修復 streaming 相關的問題
   */
  private fixStreamingIssues(response: string): string {
    let fixed = response;
    
    // 移除尾部的省略號和不完整內容
    fixed = fixed.replace(/\.{3,}.*$/, '');
    
    // 修復常見的格式錯誤
    fixed = fixed.replace(/([^"])\s*topics\s*:/g, '$1,"topics":'); // 修復缺少逗號的 topics
    fixed = fixed.replace(/([^"])\s*id\s*:/g, '$1,"id":'); // 修復缺少逗號的 id
    fixed = fixed.replace(/([^"])\s*name\s*:/g, '$1,"name":'); // 修復缺少逗號的 name
    
    // 修復缺少引號的屬性名
    fixed = fixed.replace(/([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:/g, '$1"$2":');
    
    // 修復缺少引號的字符串值
    fixed = fixed.replace(/:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])/g, ':"$1"$2');
    
    // 移除多餘的逗號
    fixed = fixed.replace(/,(\s*[}\]])/g, '$1');
    
    // 針對陣列格式的特殊修復
    if (fixed.trim().startsWith('[')) {
      // 修復陣列中物件之間缺少逗號的問題
      fixed = fixed.replace(/\}(\s*)\{/g, '},$1{');
      
      // 修復陣列結尾可能缺少的 ]
      if (!fixed.trim().endsWith(']')) {
        // 檢查是否有未閉合的陣列
        const openBrackets = (fixed.match(/\[/g) || []).length;
        const closeBrackets = (fixed.match(/\]/g) || []).length;
        if (openBrackets > closeBrackets) {
          fixed = fixed.trim() + ']';
        }
      }
      
      // 修復陣列中最後一個物件可能不完整的問題
      const lastObjectMatch = fixed.match(/\{[^{}]*$/);
      if (lastObjectMatch) {
        // 找到最後一個完整的物件
        const lastCompleteObject = fixed.match(/\{[^{}]*\}/g);
        if (lastCompleteObject && lastCompleteObject.length > 0) {
          const lastComplete = lastCompleteObject[lastCompleteObject.length - 1];
          const lastCompleteIndex = fixed.lastIndexOf(lastComplete);
          // 截取到最後一個完整物件
          fixed = fixed.substring(0, lastCompleteIndex + lastComplete.length) + ']';
        }
      }
    }
    
    console.log('   Applied streaming-specific fixes');
    return fixed;
  }
  
  /**
   * 修復結構性問題
   */
  private fixStructuralIssues(response: string): string {
    let fixed = response;
    
    // 計算括號平衡
    const openBraces = (fixed.match(/\{/g) || []).length;
    const closeBraces = (fixed.match(/\}/g) || []).length;
    const openBrackets = (fixed.match(/\[/g) || []).length;
    const closeBrackets = (fixed.match(/\]/g) || []).length;
    
    console.log(`   Bracket analysis: {${openBraces}:${closeBraces}}, [${openBrackets}:${closeBrackets}]`);
    
    // 修復大括號不平衡
    if (openBraces > closeBraces) {
      const missingBraces = openBraces - closeBraces;
      fixed = fixed + '}'.repeat(missingBraces);
      console.log(`   Fixed braces: added ${missingBraces} closing braces`);
    }
    
    // 修復方括號不平衡
    if (openBrackets > closeBrackets) {
      const missingBrackets = openBrackets - closeBrackets;
      fixed = fixed + ']'.repeat(missingBrackets);
      console.log(`   Fixed brackets: added ${missingBrackets} closing brackets`);
    }
    
    return fixed;
  }
  


  /**
   * 找到最後一個有效的 JSON 結構
   */
  private findLastValidJson(response: string): string {
    console.log('   🔍 Searching for last valid JSON structure...');
    
    // 針對陣列格式的回應進行特殊處理
    if (response.trim().startsWith('[')) {
      console.log('   Detected array format, searching for complete objects...');
      
      // 尋找所有完整的物件
      const objectMatches = response.match(/\{[^{}]*\}/g);
      if (objectMatches && objectMatches.length > 0) {
        console.log(`   Found ${objectMatches.length} complete objects`);
        
        // 找到最後一個完整物件的位置
        const lastObject = objectMatches[objectMatches.length - 1];
        const lastObjectIndex = response.lastIndexOf(lastObject);
        
        // 檢查這個物件是否完整（包含所有必要屬性）
        if (this.isCompleteObject(lastObject)) {
          // 截取到最後一個完整物件，並補上陣列結尾
          const result = response.substring(0, lastObjectIndex + lastObject.length) + ']';
          console.log('   ✅ Truncated to last complete object and closed array');
          return result;
        } else {
          // 如果最後一個物件不完整，嘗試前一個
          if (objectMatches.length > 1) {
            const secondLastObject = objectMatches[objectMatches.length - 2];
            const secondLastIndex = response.lastIndexOf(secondLastObject, lastObjectIndex - 1);
            if (this.isCompleteObject(secondLastObject)) {
              const result = response.substring(0, secondLastIndex + secondLastObject.length) + ']';
              console.log('   ✅ Truncated to second-to-last complete object and closed array');
              return result;
            }
          }
        }
      }
      
      // 如果沒有找到完整物件，嘗試修復最後一個不完整的物件
      const lastIncompleteObject = this.findLastIncompleteObject(response);
      if (lastIncompleteObject) {
        const result = lastIncompleteObject + ']';
        console.log('   ✅ Repaired last incomplete object and closed array');
        return result;
      }
    }
    
    // 尋找最後一個完整的物件（通用情況）
    const objectMatches = response.match(/\{[^{}]*\}/g);
    if (objectMatches && objectMatches.length > 0) {
      const lastObject = objectMatches[objectMatches.length - 1];
      const lastObjectIndex = response.lastIndexOf(lastObject);
      
      // 檢查這個物件是否在陣列中
      const beforeObject = response.substring(0, lastObjectIndex);
      const openBrackets = (beforeObject.match(/\[/g) || []).length;
      const closeBrackets = (beforeObject.match(/\]/g) || []).length;
      
      if (openBrackets > closeBrackets) {
        // 物件在陣列中，需要補上陣列結尾
        const result = response.substring(0, lastObjectIndex + lastObject.length) + ']';
        console.log('   Found last valid object in array, truncating there');
        return result;
      }
    }
    
    // 如果沒有找到完整物件，嘗試找到最後一個完整的陣列
    const arrayMatches = response.match(/\[[^\[\]]*\]/g);
    if (arrayMatches && arrayMatches.length > 0) {
      const lastArray = arrayMatches[arrayMatches.length - 1];
      const lastArrayIndex = response.lastIndexOf(lastArray);
      const result = response.substring(0, lastArrayIndex + lastArray.length);
      console.log('   Found last valid array, truncating there');
      return result;
    }
    
    // 最後的嘗試：如果開頭是陣列，至少補上結尾
    if (response.trim().startsWith('[')) {
      const result = response.trim() + ']';
      console.log('   Array starts but never ends, adding closing bracket');
      return result;
    }
    
    // 如果都失敗了，返回原始內容
    console.log('   Could not find valid JSON structure, returning original');
    return response;
  }
  
  /**
   * 檢查物件是否完整（包含必要的屬性）
   */
  private isCompleteObject(objectStr: string): boolean {
    try {
      const obj = JSON.parse(objectStr);
      // 檢查是否有 id 和 topics 屬性
      return obj.hasOwnProperty('id') && obj.hasOwnProperty('topics') && Array.isArray(obj.topics);
    } catch {
      return false;
    }
  }
  
  /**
   * 尋找並修復最後一個不完整的物件
   */
  private findLastIncompleteObject(response: string): string | null {
    // 尋找最後一個不完整的物件（以 { 開始但沒有 } 結尾）
    const lastOpenBrace = response.lastIndexOf('{');
    if (lastOpenBrace === -1) return null;
    
    // 從最後一個 { 開始，尋找可能的物件結構
    const afterOpenBrace = response.substring(lastOpenBrace);
    
    // 嘗試找到 id 屬性
    const idMatch = afterOpenBrace.match(/"id"\s*:\s*"([^"]+)"/);
    if (!idMatch) return null;
    
    // 嘗試找到 topics 屬性
    const topicsMatch = afterOpenBrace.match(/"topics"\s*:\s*\[/);
    if (!topicsMatch) return null;
    
    // 嘗試找到 name 屬性
    const nameMatch = afterOpenBrace.match(/"name"\s*:\s*"([^"]+)"/);
    if (!nameMatch) return null;
    
    // 如果找到了基本結構，嘗試補上缺失的部分
    const nameEndIndex = afterOpenBrace.indexOf(nameMatch[0]) + nameMatch[0].length;
    const beforeNameEnd = afterOpenBrace.substring(0, nameEndIndex);
    
    // 補上缺失的結尾
    let result = beforeNameEnd;
    
    // 補上 topics 陣列結尾
    if (!result.includes(']')) {
      result += ']';
    }
    
    // 補上物件結尾
    if (!result.includes('}')) {
      result += '}';
    }
    
    console.log('   Repaired incomplete object structure');
    return result;
  }


}

function validateResponse(response: string): boolean {
  try {
    JSON.parse(response);
    return true;
  } catch {
    return false;
  }
}
