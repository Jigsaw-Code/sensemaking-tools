#!/usr/bin/env node

import OpenAI from 'openai';
import { getEnvVar, getRequiredEnvVar } from '../src/utils/env_loader';
import { align_response_text } from '../src/utils/align_response';

async function main() {
  // 檢查命令列參數
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error('請提供一個 prompt 參數');
    console.error('使用方式: npx ts-node simple_ai_prompt.ts "你的問題"');
    process.exit(1);
  }

  const prompt = args[0];

  // 檢查必要的環境變數
  const apiKey = getRequiredEnvVar('OPENROUTER_API_KEY');
  const model = getRequiredEnvVar('OPENROUTER_MODEL');
  const baseURL = getEnvVar('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1');

  if (!apiKey) {
    console.error('錯誤: 未設定 OPENROUTER_API_KEY 環境變數');
    console.error('請在 .env 檔案中設定你的 OpenRouter API 金鑰');
    process.exit(1);
  }

  if (!model) {
    console.error('錯誤: 未設定 OPENROUTER_MODEL 環境變數');
    console.error('請在 .env 檔案中設定要使用的模型');
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

    // 發送請求
    const completion = await openai.chat.completions.create({
      model,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
      max_tokens: 1000,
      temperature: 0.7,
    });

    // 輸出結果

    // console.log(completion);
    
    console.log(completion.choices[0]?.message);

    const response_msg = completion.choices[0]?.message;
     
    //console.log((response_msg as any).reasoning);

    const response_text = align_response_text(model, response_msg);

    if (response_text) {
      console.log('AI 回應:');
      console.log(response_text);
    } else {
      console.log('未收到回應，或回應解析失敗');
    }

  } catch (error) {
    console.error('發生錯誤:');
    if (error instanceof Error) {
      console.error(error.message);
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
