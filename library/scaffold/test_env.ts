#!/usr/bin/env node

import * as dotenv from 'dotenv';
import * as path from 'path';

console.log('=== 環境變數測試 ===\n');

// 載入環境變數
console.log('1. 載入 .env 檔案...');
const result = dotenv.config({ path: path.join(__dirname, '../../.env') });

if (result.error) {
  console.error('❌ 載入 .env 檔案失敗:', result.error.message);
} else {
  console.log('✅ .env 檔案載入成功');
}

console.log('\n2. 檢查環境變數:');
console.log(`   OPENROUTER_API_KEY: ${process.env.OPENROUTER_API_KEY ? '已設定' : '未設定'}`);
console.log(`   OPENROUTER_MODEL: ${process.env.OPENROUTER_MODEL || '未設定'}`);
console.log(`   OPENROUTER_BASE_URL: ${process.env.OPENROUTER_BASE_URL || '未設定'}`);
console.log(`   DEFAULT_OPENROUTER_PARALLELISM: ${process.env.DEFAULT_OPENROUTER_PARALLELISM || '未設定'}`);

console.log('\n3. 檢查 API 端點:');
const baseURL = process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1';
console.log(`   最終使用的 API 端點: ${baseURL}`);

console.log('\n4. 檢查是否為 Cloudflare 端點:');
if (baseURL.includes('cloudflare') || baseURL.includes('cf-ray')) {
  console.log('⚠️  警告: 檢測到 Cloudflare 端點，這可能不是正確的 OpenRouter API');
} else {
  console.log('✅ API 端點看起來正確');
}

console.log('\n=== 測試完成 ===');
