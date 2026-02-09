import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';

/**
 * 檢測當前運行環境
 * @returns 'node' | 'worker' | 'unknown'
 */
function detectEnvironment(): 'node' | 'worker' | 'unknown' {
  // 檢查是否在 Node.js 環境中
  if (typeof process !== 'undefined' && process.env && process.versions && process.versions.node) {
    return 'node';
  }
  
  // 檢查是否在 Cloudflare Workers 環境中
  if (typeof globalThis !== 'undefined' && (globalThis as any).__CLOUDFLARE_WORKER__) {
    return 'worker';
  }
  
  // 檢查是否在瀏覽器環境中
  if (typeof globalThis !== 'undefined' && typeof (globalThis as any).window !== 'undefined') {
    return 'worker'; // 瀏覽器環境也使用 globalThis 方式
  }
  
  return 'unknown';
}

/**
 * Load environment variables with priority: system env > .env file
 * This ensures npm package compatibility while maintaining development convenience
 */
export function loadEnvironmentVariables(): void {
  const env = detectEnvironment();
  
  if (env === 'node') {
    // Node.js 環境：使用傳統的 .env 文件載入方式
    if (process.env.NODE_ENV !== 'production') {
      try {
        // 檢查多個可能的 .env 檔案路徑
        const possiblePaths = [
          '.env',
          path.resolve(process.cwd(), '.env'),
          path.resolve(__dirname, '../../../.env'),
          path.resolve(__dirname, '../../.env'),
          path.resolve(__dirname, '../.env')
        ];
        
        for (const envPath of possiblePaths) {
          if (fs.existsSync(envPath)) {
            try {
              dotenv.config({ path: envPath });
              console.log(`📁 載入環境變數檔案: ${envPath}`);
              break;
            } catch {
              console.debug('dotenv not available, using system environment variables only');
              break;
            }
          }
        }
      } catch {
        // 如果檔案系統操作失敗，靜默忽略
        console.debug('File system operations failed, using system environment variables only');
      }
    }
  } else if (env === 'worker') {
    // Cloudflare Workers 環境：環境變量已經通過 wrangler 配置
    console.log('📁 Environment variables loaded via Wrangler configuration');
  } else {
    console.log('📁 Unknown environment, using fallback environment variable access');
  }
}

/**
 * Get environment variable with fallback to .env file
 * @param key Environment variable key
 * @param defaultValue Default value if not found
 * @returns Environment variable value
 */
export function getEnvVar(key: string, defaultValue?: string): string | undefined {
  const env = detectEnvironment();
  
  if (env === 'node') {
    // Node.js 環境：優先讀取系統環境變數
    if (process.env[key]) {
      return process.env[key];
    }
    
    // 如果系統環境變數不存在，嘗試載入 .env 檔案
    loadEnvironmentVariables();
    
    // 再次檢查系統環境變數（可能已經被 .env 載入）
    return process.env[key] || defaultValue;
  } else if (env === 'worker') {
    // Cloudflare Workers 環境：通過 globalThis 訪問
    return (globalThis as any)[key] || defaultValue;
  } else {
    // 未知環境：嘗試多種方式
    return (globalThis as any)[key] || process.env?.[key] || defaultValue;
  }
}

/**
 * Get required environment variable
 * @param key Environment variable key
 * @returns Environment variable value
 * @throws Error if environment variable is not set
 */
export function getRequiredEnvVar(key: string): string {
  const value = getEnvVar(key);
  if (!value) {
    throw new Error(`${key} environment variable is required`);
  }
  return value;
}
