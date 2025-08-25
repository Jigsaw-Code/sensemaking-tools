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
        
        // 顯示前幾個chunks的原始內容
        /* if (context.chunkCount <= 5) {
          console.log(`   Raw chunk ${context.chunkCount}: "${decodedChunk}"`);
        } */
        
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
          /*  else if (line.trim()) {
            // 顯示非data行的內容
            if (context.chunkCount <= 10) {
              console.log(`   Non-data line: "${line}"`);
            }
          } */
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
   * 檢測是否為包含 JSON 的混合格式
   */
  private isMixedFormat(response: string): boolean {
    const trimmed = response.trim();
    
    // 檢查是否包含 JSON 代碼區塊
    const hasJsonBlock = /```json\s*[\s\S]*?```/g.test(trimmed);
    const hasCodeBlock = /```\s*[\s\S]*?```/g.test(trimmed);
    
    // 檢查是否包含 JSON 結構（大括號或方括號）
    const hasJsonStructure = /\{[\s\S]*\}|\[[\s\S]*\]/g.test(trimmed);
    
    // 檢查是否為純文本描述（包含中文、英文等自然語言）
    const hasNaturalLanguage = /[\u4e00-\u9fff\u3400-\u4dbf]|[a-zA-Z]{3,}/g.test(trimmed);
    
    // 如果同時包含自然語言和 JSON 結構，則為混合格式
    if (hasNaturalLanguage && (hasJsonBlock || hasCodeBlock || hasJsonStructure)) {
      console.log('   🔍 Detected mixed format: natural language + JSON structure');
      return true;
    }
    
    // 如果包含 JSON 代碼區塊，也視為混合格式
    if (hasJsonBlock || hasCodeBlock) {
      console.log('   🔍 Detected code block format');
      return true;
    }
    
    return false;
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
    
    // 智能格式檢測：檢查是否為混合格式（文本 + JSON）
    if (this.isMixedFormat(processedResponse)) {
      console.log('   📝 Detected mixed format, extracting JSON blocks...');
      const extractedJson = this.extractJsonFromMixedContent(processedResponse);
      if (extractedJson) {
        console.log('   ✅ Successfully extracted JSON from mixed content');
        return extractedJson;
      }
    }
    
    // 檢查並修復 JSON 完整性
    processedResponse = this.fixIncompleteJson(processedResponse);
    
    const finalResponse = processedResponse.trim();
    console.log('   Final processed length:', finalResponse.length);
    console.log('   Final processed content:', finalResponse);
    
    return finalResponse;
  }

  /**
   * 從混合內容中提取 JSON
   */
  private extractJsonFromMixedContent(content: string): string | null {
    // 優先尋找 ```json ... ``` 區塊
    const jsonBlockRegex = /```json\s*([\s\S]*?)\s*```/g;
    const jsonMatches = [...content.matchAll(jsonBlockRegex)];
    
    if (jsonMatches.length > 0) {
      console.log(`   🔍 Found ${jsonMatches.length} JSON code blocks`);
      const jsonContent = jsonMatches[0][1].trim();
      if (this.isValidJson(jsonContent)) {
        console.log(`   ✅ Extracted valid JSON from json block`);
        return jsonContent;
      }
    }
    
    // 尋找 ``` ... ``` 區塊（沒有 json 標籤）
    const codeBlockRegex = /```\s*([\s\S]*?)\s*```/g;
    const codeMatches = [...content.matchAll(codeBlockRegex)];
    
    if (codeMatches.length > 0) {
      console.log(`   🔍 Found ${codeMatches.length} code blocks`);
      for (const match of codeMatches) {
        const codeContent = match[1].trim();
        if (this.isValidJson(codeContent)) {
          console.log(`   ✅ Found valid JSON in code block`);
          return codeContent;
        }
      }
    }
    
    // 尋找內嵌的 JSON 結構
    const jsonObjectRegex = /\{[\s\S]*\}/g;
    const jsonArrayRegex = /\[[\s\S]*\]/g;
    
    const objectMatches = [...content.matchAll(jsonObjectRegex)];
    const arrayMatches = [...content.matchAll(jsonArrayRegex)];
    
    if (objectMatches.length > 0 || arrayMatches.length > 0) {
      console.log(`   🔍 Found ${objectMatches.length} JSON objects and ${arrayMatches.length} JSON arrays`);
      
      // 返回最長的匹配項
      const allMatches = [...objectMatches, ...arrayMatches];
      const longestMatch = allMatches.reduce((longest, current) => 
        current[0].length > longest[0].length ? current : longest
      );
      
      if (this.isValidJson(longestMatch[0])) {
        console.log(`   ✅ Found valid JSON structure`);
        return longestMatch[0];
      }
    }
    
    console.log('   ❌ No valid JSON found in mixed content');
    return null;
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
    
    // 進行結構性修復（括號對齊）
    fixedResponse = this.fixStructuralIssues(fixedResponse);
    
    // 嘗試解析修復後的內容
    try {
      JSON.parse(fixedResponse);
      console.log('   ✅ JSON validation passed after structural repair');
      return fixedResponse;
    } catch {
      console.log('   ❌ Still invalid after structural repair, attempting smart truncation...');
    }
    
    // 智能截斷：在括號對齊的基礎上進行截斷
    fixedResponse = this.smartTruncateJson(fixedResponse);
    
    // 最終驗證
    try {
      JSON.parse(fixedResponse);
      console.log('   ✅ JSON validation passed after smart truncation');
      return fixedResponse;
    } catch (error) {
      console.log('   ❌ JSON validation failed after all repair attempts');
      console.log('   Final repair attempt failed:', error);
      
      // 最後的嘗試：強制修復成有效的 JSON
      return this.forceFixJson(fixedResponse);
    }
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
   * 智能截斷 JSON：在括號對齊的基礎上進行截斷
   */
  private smartTruncateJson(response: string): string {
    console.log('   🔍 Starting smart JSON truncation...');
    
    // 分析括號結構
    const bracketAnalysis = this.analyzeBracketStructure(response);
    console.log('   Bracket analysis:', bracketAnalysis);
    
    // 如果括號已經平衡，嘗試找到最後一個完整的物件或陣列
    if (bracketAnalysis.isBalanced) {
      console.log('   Brackets are balanced, looking for complete structures...');
      return this.findCompleteJsonStructure(response);
    }
    
    // 如果括號不平衡，先嘗試修復
    const fixedResponse = this.balanceBrackets(response, bracketAnalysis);
    
    // 嘗試解析修復後的內容
    try {
      JSON.parse(fixedResponse);
      console.log('   ✅ Successfully balanced brackets and created valid JSON');
      return fixedResponse;
    } catch {
      console.log('   ❌ Still invalid after bracket balancing, attempting structure completion...');
    }
    
    // 如果修復後仍然無效，嘗試完成結構
    return this.completeJsonStructure(fixedResponse);
  }
  
  /**
   * 分析括號結構
   */
  private analyzeBracketStructure(response: string): {
    openBraces: number;
    closeBraces: number;
    openBrackets: number;
    closeBrackets: number;
    isBalanced: boolean;
    braceBalance: number;
    bracketBalance: number;
  } {
    const openBraces = (response.match(/\{/g) || []).length;
    const closeBraces = (response.match(/\}/g) || []).length;
    const openBrackets = (response.match(/\[/g) || []).length;
    const closeBrackets = (response.match(/\]/g) || []).length;
    
    const braceBalance = openBraces - closeBraces;
    const bracketBalance = openBrackets - closeBrackets;
    const isBalanced = braceBalance === 0 && bracketBalance === 0;
    
    return {
      openBraces,
      closeBraces,
      openBrackets,
      closeBrackets,
      isBalanced,
      braceBalance,
      bracketBalance
    };
  }
  
  /**
   * 平衡括號
   */
  private balanceBrackets(response: string, analysis: ReturnType<typeof this.analyzeBracketStructure>): string {
    let balanced = response;
    
    // 補上缺少的右大括號
    if (analysis.braceBalance > 0) {
      balanced += '}'.repeat(analysis.braceBalance);
      console.log(`   Added ${analysis.braceBalance} closing braces`);
    }
    
    // 補上缺少的右方括號
    if (analysis.bracketBalance > 0) {
      balanced += ']'.repeat(analysis.bracketBalance);
      console.log(`   Added ${analysis.bracketBalance} closing brackets`);
    }
    
    return balanced;
  }
  
  /**
   * 找到完整的 JSON 結構
   */
  private findCompleteJsonStructure(response: string): string {
    // 尋找最後一個完整的物件
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
        console.log('   Found last complete object in array, truncating there');
        return result;
      }
    }
    
    // 如果沒有找到完整物件，嘗試找到最後一個完整的陣列
    const arrayMatches = response.match(/\[[^\[\]]*\]/g);
    if (arrayMatches && arrayMatches.length > 0) {
      const lastArray = arrayMatches[arrayMatches.length - 1];
      const lastArrayIndex = response.lastIndexOf(lastArray);
      const result = response.substring(0, lastArrayIndex + lastArray.length);
      console.log('   Found last complete array, truncating there');
      return result;
    }
    
    // 最後的嘗試：如果開頭是陣列，至少補上結尾
    if (response.trim().startsWith('[')) {
      const result = response.trim() + ']';
      console.log('   Array starts but never ends, adding closing bracket');
      return result;
    }
    
    // 如果都失敗了，返回原始內容
    console.log('   Could not find complete JSON structure, returning original');
    return response;
  }
  
  /**
   * 完成 JSON 結構
   */
  private completeJsonStructure(response: string): string {
    console.log('   🔧 Completing JSON structure...');
    
    let completed = response;
    
    // 如果是以陣列開頭，確保陣列結構完整
    if (response.trim().startsWith('[')) {
      // 找到最後一個完整的物件
      const objectMatches = response.match(/\{[^{}]*\}/g);
      if (objectMatches && objectMatches.length > 0) {
        const lastObject = objectMatches[objectMatches.length - 1];
        const lastObjectIndex = response.lastIndexOf(lastObject);
        
        // 截斷到最後一個完整物件，然後補上陣列結尾
        completed = response.substring(0, lastObjectIndex + lastObject.length) + ']';
        console.log('   Completed array structure with last valid object');
      } else {
        // 沒有完整物件，補上空陣列
        completed = '[]';
        console.log('   No valid objects found, returning empty array');
      }
    }
    // 如果是以物件開頭，確保物件結構完整
    else if (response.trim().startsWith('{')) {
      // 找到最後一個完整的屬性
      const propertyMatches = response.match(/"\w+"\s*:\s*[^,}]+/g);
      if (propertyMatches && propertyMatches.length > 0) {
        const lastProperty = propertyMatches[propertyMatches.length - 1];
        const lastPropertyIndex = response.lastIndexOf(lastProperty);
        
        // 截斷到最後一個完整屬性，然後補上物件結尾
        completed = response.substring(0, lastPropertyIndex + lastProperty.length) + '}';
        console.log('   Completed object structure with last valid property');
      } else {
        // 沒有完整屬性，補上空物件
        completed = '{}';
        console.log('   No valid properties found, returning empty object');
      }
    }
    
    return completed;
  }
  
  /**
   * 強制修復 JSON（最後手段）
   */
  private forceFixJson(response: string): string {
    console.log('   🚨 Force fixing JSON as last resort...');
    
    const trimmed = response.trim();
    
    // 如果開頭是陣列，強制補上結尾
    if (trimmed.startsWith('[')) {
      // 尋找最後一個看起來完整的物件
      const lastBraceIndex = trimmed.lastIndexOf('}');
      if (lastBraceIndex > 0) {
        const result = trimmed.substring(0, lastBraceIndex + 1) + ']';
        console.log('   Force fixed array structure');
        return result;
      } else {
        console.log('   Force returning empty array');
        return '[]';
      }
    }
    
    // 如果開頭是物件，強制補上結尾
    if (trimmed.startsWith('{')) {
      const lastBraceIndex = trimmed.lastIndexOf('}');
      if (lastBraceIndex > 0) {
        const result = trimmed.substring(0, lastBraceIndex + 1);
        console.log('   Force fixed object structure');
        return result;
      } else {
        console.log('   Force returning empty object');
        return '{}';
      }
    }
    
    // 如果都不符合，嘗試包裝成陣列
    console.log('   Force wrapping in array');
    return `[${trimmed}]`;
  }

  /**
   * 驗證 JSON 是否有效
   */
  private isValidJson(json: string): boolean {
    try {
      JSON.parse(json);
      return true;
    } catch (e) {
      console.warn('   Invalid JSON in extracted block:', e);
      return false;
    }
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
