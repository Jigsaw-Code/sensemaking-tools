#!/usr/bin/env node

import OpenAI from 'openai';
import * as dotenv from 'dotenv';
import * as path from 'path';
import { Type, TSchema } from '@sinclair/typebox';
import { TypeCompiler } from '@sinclair/typebox/compiler';

// 載入環境變數
dotenv.config({ path: path.join(__dirname, '../../.env') });

// 定義 JSON Schema 類型
interface OpenRouterRequest {
  model: string;
  messages: Array<{
    role: 'user' | 'assistant' | 'system';
    content: string;
  }>;
  response_format?: {
    type: 'json_schema';
    json_schema: {
      name: string;
      strict: boolean;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      schema: any;
    };
  };
  max_tokens?: number;
  temperature?: number;
}

// 範例 JSON Schema - 可以根據需求修改
const EXAMPLE_SCHEMA = Type.Object({
  name: Type.String({ description: "The name of the person" }),
  age: Type.Number({ description: "The age of the person" }),
  interests: Type.Array(Type.String(), { description: "List of interests" }),
  location: Type.Object({
    city: Type.String({ description: "City name" }),
    country: Type.String({ description: "Country name" })
  }),
  isActive: Type.Boolean({ description: "Whether the person is active" })
});


// 建立 OpenRouter 請求的 JSON Schema

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function createJsonSchemaRequest(schema: TSchema, schemaName: string): any {
  return {
    type: 'json_schema',
    json_schema: {
      name: schemaName,
      strict: true,
      schema: schema
    }
  };
}

// 驗證 JSON 回應是否符合 Schema
function validateJsonResponse(response: string, schema: TSchema): boolean {
  try {
    const parsedResponse = JSON.parse(response);
    console.log('解析的 JSON:', parsedResponse);

    // 使用 TypeBox 編譯器進行嚴格的 schema 驗證
    const checker = TypeCompiler.Compile(schema);
    const isValid = checker.Check(parsedResponse);
    
    if (!isValid) {
      const errors = checker.Errors(parsedResponse);
      console.error('Schema 驗證失敗:');
      let errorIndex = 0;
      for (const error of errors) {
        errorIndex++;
        console.error(`  錯誤 ${errorIndex}: ${error.message}`);
        console.error(`    路徑: ${error.path}`);
        console.error(`    值: ${error.value}`);
      }
      return false;
    }
    
    console.log('✓ Schema 驗證通過');
    return true;
  } catch (e) {
    console.error(`Model returned a non-JSON response:\n${response}\n${e}`);
    return false;
  }
}

async function main() {
  // 檢查命令列參數
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error('請提供一個 prompt 參數');
    console.error('使用方式: npx ts-node JSON_ai_prompt.ts "你的問題"');
    process.exit(1);
  }

  const prompt = args[0];

  // 檢查必要的環境變數
  const apiKey = process.env.OPENROUTER_API_KEY;
  const model = process.env.OPENROUTER_MODEL;
  const baseURL = process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1';

  if (!apiKey) {
    console.error('錯誤: 未設定 OPENROUTER_API_KEY 環境變數');
    console.error('請在 library/.env 檔案中設定你的 OpenRouter API 金鑰');
    process.exit(1);
  }

  if (!model) {
    console.error('錯誤: 未設定 OPENROUTER_MODEL 環境變數');
    console.error('請在 library/.env 檔案中設定要使用的模型');
    process.exit(1);
  }

  console.log(`使用模型: ${model}`);
  console.log(`API 端點: ${baseURL}`);
  console.log(`Prompt: ${prompt}`);
  console.log('---');

  try {
    // 建立 OpenAI 客戶端
    const openai = new OpenAI({
      apiKey,
      baseURL,
    });

    // 建立結構化輸出的請求
    const request: OpenRouterRequest = {
      model,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
      response_format: createJsonSchemaRequest(EXAMPLE_SCHEMA, 'example'),
      max_tokens: 1000,
      temperature: 0.1, // 降低溫度以獲得更一致的結構化輸出
    };

    console.log('發送結構化輸出請求...');
    console.log('JSON Schema:', JSON.stringify(request.response_format, null, 2));

    // 發送請求
    const completion = await openai.chat.completions.create({
      model: request.model,
      messages: request.messages,
      max_tokens: request.max_tokens,
      temperature: request.temperature,
      response_format: request.response_format,
    });

    const responseText = completion.choices[0]?.message?.content;

    if (!responseText) {
      console.error('未收到回應');
      return;
    }

    console.log('原始回應:');
    console.log(responseText);
    console.log('---');

    // 驗證 JSON 回應
    if (validateJsonResponse(responseText, EXAMPLE_SCHEMA)) {
      try {
        const parsedResponse = JSON.parse(responseText);
        console.log('解析後的 JSON 資料:');
        console.log(JSON.stringify(parsedResponse, null, 2));
        
        // 可以根據 schema 進行類型檢查
        console.log('---');
        console.log('資料摘要:');
        console.log(`姓名: ${parsedResponse.name}`);
        console.log(`年齡: ${parsedResponse.age}`);
        console.log(`興趣: ${parsedResponse.interests?.join(', ')}`);
        console.log(`城市: ${parsedResponse.location?.city}`);
        console.log(`國家: ${parsedResponse.location?.country}`);
        console.log(`活躍狀態: ${parsedResponse.isActive}`);
        
      } catch (parseError) {
        console.error('JSON 解析失敗:', parseError);
      }
    } else {
      console.error('回應不符合預期的 JSON Schema');
    }

  } catch (error) {
    console.error('發生錯誤:');
    if (error instanceof Error) {
      console.error(error.message);
      
      // 檢查是否為 OpenRouter 特定錯誤
      if (error.message.includes('structured output') || error.message.includes('json_schema')) {
        console.error('\n提示: 此模型可能不支援結構化輸出');
        console.error('請嘗試使用支援 structured outputs 的模型，如:');
        console.error('- openai/gpt-4o');
        console.error('- anthropic/claude-3.5-sonnet');
        console.error('- google/gemini-pro');
      }
    } else {
      console.error(error);
    }
    process.exit(1);
  }
}

// 執行主程式
main().catch((error) => {
  console.error('程式執行失敗:', error);
  process.exit(1);
});
