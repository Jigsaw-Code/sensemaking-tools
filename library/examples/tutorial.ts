#!/usr/bin/env node

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

// Sensemaker scaffold example using OpenRouterModel instead of VertexModel
// This demonstrates how to use the new OpenRouter integration

import { Sensemaker } from '../src/sensemaker';
import { OpenRouterModel } from '../src/models/openrouter_model';
import { SummarizationType, Comment, VoteTally } from '../src/types';
import { getEnvVar } from '../src/utils/env_loader';
import * as fs from 'fs';

// CSV 讀取函數
function getCommentsFromCsv(csvPath: string): Comment[] {
  try {
    const csvContent = fs.readFileSync(csvPath, 'utf-8');
    const lines = csvContent.split('\n').filter(line => line.trim());
    
    const comments: Comment[] = [];
    
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(val => val.trim().replace(/^"|"$/g, ''));
      const commentId = values[0];
      const commentText = values[1];
      const agrees = parseInt(values[3]) || 0;
      const disagrees = parseInt(values[4]) || 0;
      const passes = parseInt(values[5]) || 0;
      
      if (commentId && commentText) {
        comments.push({
          id: commentId,
          text: commentText,
          voteInfo: { "group1": new VoteTally(agrees, disagrees, passes) }
        });
      }
    }
    
    return comments;
  } catch (error) {
    console.error(`❌ 讀取 CSV 檔案失敗: ${error}`);
    return [];
  }
}

async function main() {
  try {
    console.log('🚀 啟動 Sensemaker 腳本...\n');

    // 檢查環境變數
    if (!getEnvVar('OPENROUTER_API_KEY')) {
      throw new Error('❌ 缺少 OPENROUTER_API_KEY 環境變數');
    }

    console.log('✅ 環境變數載入成功');
    console.log(`🔑 API 金鑰: ${getEnvVar('OPENROUTER_API_KEY') ? '已設定' : '未設定'}`);
    console.log(`🤖 模型: ${getEnvVar('OPENROUTER_MODEL', '使用預設值')}`);
    console.log(`🌐 API 端點: ${getEnvVar('OPENROUTER_BASE_URL', '使用預設值')}`);
    console.log(`⚡ 並發限制: ${getEnvVar('DEFAULT_OPENROUTER_PARALLELISM', '使用預設值')}\n`);

    // 使用 OpenRouter 模型建立 Sensemaker 實例
    const openRouterModel = new OpenRouterModel(
      getEnvVar('OPENROUTER_API_KEY') || '',
      getEnvVar('OPENROUTER_MODEL', 'anthropic/claude-3.5-sonnet')
    );
    console.log(`✅ OpenRouter 模型建立成功`);
    console.log(`🔑 API 金鑰: ${getEnvVar('OPENROUTER_API_KEY') ? '已設定' : '未設定'}`);
    console.log(`🤖 模型: ${getEnvVar('OPENROUTER_MODEL', '使用預設值')}\n`);

    const mySensemaker = new Sensemaker({
      defaultModel: openRouterModel,
    });

    console.log('✅ Sensemaker 實例建立成功\n');

    // TODO: 從 CSV 檔案讀取評論數據
    // CSV 包含評論文字、投票計數和群組信息
    console.log('📊 準備從 CSV 檔案讀取評論數據...');
    
    // 暫時使用示例數據，等待 CSV 檔案準備好
    // 當 CSV 檔案準備好後，可以替換這個部分
    const comments: Comment[] = getCommentsFromCsv("./files/comments.csv")

    console.log(`✅ 載入 ${comments.length} 條評論\n`);

    if (comments.length === 0) {
      console.error('❌ 沒有載入任何評論');
      process.exit(1);
    }

    // 學習討論的主題並輸出
    console.log('🔍 開始學習討論主題...');
    const topics = await mySensemaker.learnTopics(
      comments,
      // 應該包含子主題:
      true,
      // 沒有現有主題:
      undefined,
      // 額外上下文:
      "This is from a conversation about Taiwan's homeschooling system and community development",
      // 主題深度:
      2,
      // 輸出語言:
      "zh-TW"
    );
    
    console.log('✅ 主題學習完成');
    console.log('📋 識別的主題:');
    console.log(JSON.stringify(topics, null, 2));
    console.log();

    // 總結對話並以 Markdown 格式輸出結果
    console.log('📝 開始總結對話...');
    const summary = await mySensemaker.summarize(
      comments,
      SummarizationType.AGGREGATE_VOTE,
      topics,
      // 額外上下文:
      "This is from a conversation about Taiwan's homeschooling system and community development",
      // 輸出語言:
      "zh-TW"
    );
    
    console.log('✅ 對話總結完成');
    console.log('📄 Markdown 格式的總結:');
    console.log('---');
    console.log(summary.getText("MARKDOWN"));
    console.log('---');

    // 也可以輸出 XML 格式
    // console.log('\n📄 XML 格式的總結:');
    // console.log('---');
    // console.log(summary.getText("XML"));
    // console.log('---');

    console.log('\n🎉 腳本執行完成！');

  } catch (error) {
    console.error('❌ 腳本執行失敗:', error);
    process.exit(1);
  }
}

// 執行主函數
if (require.main === module) {
  main();
}

export { main };
