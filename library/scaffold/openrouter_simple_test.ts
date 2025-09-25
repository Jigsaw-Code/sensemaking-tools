#!/usr/bin/env node

import { OpenRouterModel } from '../src/models/openrouter_model';
import { getRequiredEnvVar } from '../src/utils/env_loader';

async function testOpenRouterModel() {
  try {
    console.log('🚀 開始測試 OpenRouter 模型...\n');

    // 檢查環境變數
    const apiKey = getRequiredEnvVar('OPENROUTER_API_KEY');
    console.log('✅ API 金鑰已設定');

    const model = getRequiredEnvVar('OPENROUTER_MODEL');
    console.log(`✅ 使用模型: ${model}`);
    console.log(`✅ API 金鑰: ${apiKey.substring(0, 8)}...\n`);

    // 創建模型實例
    console.log('1. 創建 OpenRouter 模型實例...');
    const openRouterModel = new OpenRouterModel(apiKey, model);
    console.log('✅ 模型創建成功\n');

    // 測試文字生成
    console.log('2. 測試文字生成...');
    const prompt = "請用一句話描述人工智慧的優點";
    console.log(`問題: ${prompt}`);
    
    const response = await openRouterModel.generateText(prompt);
    console.log(`回答: ${response}\n`);

    console.log('=== 測試完成 ===');
    console.log('✅ OpenRouter 模型實作正常工作！');

  } catch (error) {
    console.error('❌ 測試失敗:');
    if (error instanceof Error) {
      console.error(error.message);
      
      if (error.message.includes('OPENROUTER_API_KEY')) {
        console.error('\n💡 解決方案:');
        console.error('1. 在 library/.env 檔案中設定 OPENROUTER_API_KEY');
        console.error('2. 從 https://openrouter.ai/ 獲取 API 金鑰');
      } else if (error.message.includes('fetch') || error.message.includes('network')) {
        console.error('\n💡 網路問題，請檢查:');
        console.error('1. 網路連線是否正常');
        console.error('2. API 金鑰是否有效');
        console.error('3. 是否達到 API 使用限制');
      }
    } else {
      console.error(error);
    }
  }
}

// 執行測試
if (require.main === module) {
  testOpenRouterModel().catch((error) => {
    console.error('程式執行失敗:', error);
    process.exit(1);
  });
}
