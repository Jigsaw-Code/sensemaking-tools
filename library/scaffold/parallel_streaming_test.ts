// 測試並行 streaming 處理，確保每個請求都有獨立的 buffer 上下文
// 避免 buffer 互相干擾的問題

import { OpenRouterModel } from "../src/models/openrouter_model";
import { getEnvVar, getRequiredEnvVar } from '../src/utils/env_loader';

async function testParallelStreaming() {
  console.log("🧪 開始測試並行 streaming 處理...");
  
  // 創建 OpenRouter 模型實例
  const apiKey = getRequiredEnvVar('OPENROUTER_API_KEY');
  if (!apiKey) {
    console.error("❌ 請設定 OPENROUTER_API_KEY 環境變數");
    return;
  }
  
  const model = new OpenRouterModel(apiKey);
  
  // 創建多個並行請求
  const requests = [
    {
      id: "request-1",
      prompt: "請生成一個包含 3 個水果名稱的 JSON 陣列，格式：[{\"name\": \"水果名\"}]",
      description: "第一個請求：水果名稱"
    },
    {
      id: "request-2", 
      prompt: "請生成一個包含 3 個顏色名稱的 JSON 陣列，格式：[{\"name\": \"顏色名\"}]",
      description: "第二個請求：顏色名稱"
    },
    {
      id: "request-3",
      prompt: "請生成一個包含 3 個動物名稱的 JSON 陣列，格式：[{\"name\": \"動物名\"}]",
      description: "第三個請求：動物名稱"
    }
  ];
  
  console.log(`📡 發送 ${requests.length} 個並行請求...`);
  
  try {
    // 並行執行所有請求
    const startTime = Date.now();
    
    const promises = requests.map(async (req) => {
      console.log(`🚀 開始執行 ${req.id}: ${req.description}`);
      const start = Date.now();
      
      try {
        const response = await model.generateText(req.prompt);
        const duration = Date.now() - start;
        console.log(`✅ ${req.id} 完成 (${duration}ms): ${response.substring(0, 100)}...`);
        return { id: req.id, success: true, response, duration };
      } catch (error) {
        const duration = Date.now() - start;
        console.error(`❌ ${req.id} 失敗 (${duration}ms):`, error);
        return { id: req.id, success: false, error, duration };
      }
    });
    
    const results = await Promise.all(promises);
    const totalTime = Date.now() - startTime;
    
    console.log("\n📊 測試結果摘要:");
    console.log(`總執行時間: ${totalTime}ms`);
    
    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);
    
    console.log(`成功: ${successful.length}/${results.length}`);
    console.log(`失敗: ${failed.length}/${results.length}`);
    
    if (successful.length > 0) {
      console.log("\n✅ 成功的請求:");
      successful.forEach(r => {
        console.log(`  ${r.id}: ${r.duration}ms`);
      });
    }
    
    if (failed.length > 0) {
      console.log("\n❌ 失敗的請求:");
      failed.forEach(r => {
        console.log(`  ${r.id}: ${r.error}`);
      });
    }
    
    // 檢查是否有 buffer 干擾的跡象
    console.log("\n🔍 Buffer 干擾檢查:");
    const responses = successful.map(r => r.response);
    const uniqueResponses = new Set(responses);
    
    if (uniqueResponses.size === responses.length) {
      console.log("✅ 所有回應都是獨立的，沒有 buffer 干擾");
    } else {
      console.log("⚠️ 發現重複回應，可能存在 buffer 干擾");
      console.log("重複的回應數量:", responses.length - uniqueResponses.size);
    }
    
  } catch (error) {
    console.error("❌ 測試執行失敗:", error);
  }
}

// 執行測試
if (require.main === module) {
  testParallelStreaming().catch(console.error);
}

export { testParallelStreaming };
