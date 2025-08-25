// 測試 overview.ts 的多語言功能
// 驗證不同語言下 prompt 的生成和本地化功能

import { OverviewSummary } from "../src/tasks/summarization_subtasks/overview";
import { SummaryStats, TopicStats } from "../src/stats/summary_stats";
import { SummaryContent } from "../src/types";
import { SupportedLanguage } from "../templates/l10n/languages";
import { getOverviewOneShotPrompt, getOverviewPerTopicPrompt } from "../templates/l10n/prompts";
import { OpenRouterModel } from "../src/models/openrouter_model";

import { getRequiredEnvVar } from '../src/utils/env_loader';

// 創建測試用的模擬數據
function createMockData(): {
  summaryStats: SummaryStats;
  topicsSummary: SummaryContent;
} {
  // 創建模擬的 TopicStats
  const topicStats: TopicStats[] = [
    { 
      name: "專制很好", 
      commentCount: 30,
      summaryStats: {} as SummaryStats
    },
    { 
      name: "民主很好", 
      commentCount: 70,
      summaryStats: {} as SummaryStats
    }
  ];

  // 創建模擬的 SummaryStats
  const summaryStats = {
    commentCount: 100,
    getStatsByTopic: () => topicStats
  } as SummaryStats;

  // 創建模擬的 SummaryContent
  const topicsSummary: SummaryContent = {
    title: "Topics Summary",
    text: "This is a mock topics summary for testing purposes.",
    citations: [],
    subContents: []
  };

  return { summaryStats, topicsSummary };
}

// 測試多語言 prompt 生成
async function testMultilangPromptGeneration() {
  console.log("🧪 開始測試多語言 prompt 生成...");
  
  const testLanguages: SupportedLanguage[] = ["en", "zh-TW", "zh-CN", "fr", "es", "ja"];
  const testTopicNames = ["專制很好 (30%)", "民主很好 (70%)"];
  const testTopicName = "專制很好";
  
  console.log("\n📝 測試 one-shot prompt 生成:");
  for (const lang of testLanguages) {
    try {
      const prompt = getOverviewOneShotPrompt(lang, testTopicNames);
      console.log(`✅ ${lang}: prompt 生成成功 (長度: ${prompt.length} 字符)`);
      
      // 檢查是否包含語言特定的內容
      if (lang === "zh-TW" || lang === "zh-CN") {
        if (prompt.includes("您的工作") || prompt.includes("您应该")) {
          console.log(`   ✓ ${lang}: 包含中文內容`);
        } else {
          console.log(`   ⚠️ ${lang}: 可能缺少中文內容`);
        }
      } else if (lang === "fr") {
        if (prompt.includes("Votre travail")) {
          console.log(`   ✓ ${lang}: 包含法文內容`);
        } else {
          console.log(`   ⚠️ ${lang}: 可能缺少法文內容`);
        }
      } else if (lang === "es") {
        if (prompt.includes("Su trabajo")) {
          console.log(`   ✓ ${lang}: 包含西班牙文內容`);
        } else {
          console.log(`   ⚠️ ${lang}: 可能缺少西班牙文內容`);
        }
      } else if (lang === "ja") {
        if (prompt.includes("あなたの仕事")) {
          console.log(`   ✓ ${lang}: 包含日文內容`);
        } else {
          console.log(`   ⚠️ ${lang}: 可能缺少日文內容`);
        }
      } else if (lang === "en") {
        if (prompt.includes("Your job")) {
          console.log(`   ✓ ${lang}: 包含英文內容`);
        } else {
          console.log(`   ⚠️ ${lang}: 可能缺少英文內容`);
        }
      }
      
      // 檢查是否包含主題名稱
      if (prompt.includes("Topic A") && prompt.includes("Topic B")) {
        console.log(`   ✓ ${lang}: 包含主題名稱`);
      } else {
        console.log(`   ❌ ${lang}: 缺少主題名稱`);
      }
      
    } catch (error) {
      console.error(`❌ ${lang}: prompt 生成失敗:`, error);
    }
  }
  
  console.log("\n📝 測試 per-topic prompt 生成:");
  for (const lang of testLanguages) {
    try {
      const prompt = getOverviewPerTopicPrompt(lang, testTopicName);
      console.log(`✅ ${lang}: prompt 生成成功 (長度: ${prompt.length} 字符)`);
      
      // 檢查是否包含主題名稱
      if (prompt.includes("Topic A")) {
        console.log(`   ✓ ${lang}: 包含主題名稱`);
      } else {
        console.log(`   ❌ ${lang}: 缺少主題名稱`);
      }
      
    } catch (error) {
      console.error(`❌ ${lang}: prompt 生成失敗:`, error);
    }
  }
}

// 測試 OverviewSummary 類的多語言功能（使用真實 LLM）
async function testOverviewSummaryMultilang() {
  console.log("\n🧪 開始測試 OverviewSummary 類的多語言功能（使用真實 LLM）...");
  
  // 檢查環境變數
  const apiKey = getRequiredEnvVar('OPENROUTER_API_KEY');
  if (!apiKey) {
    console.log("⚠️ 未設定 OPENROUTER_API_KEY 環境變數，跳過 LLM 測試");
    console.log("💡 請設定環境變數：請先 .env 中設定OPENROUTER_API_KEY 環境變數");
    return;
  }
  
  const testLanguages: SupportedLanguage[] = ["en", "zh-TW", "zh-CN"];
  const mockData = createMockData();
  
  // 創建真實的 OpenRouter 模型
  const model = new OpenRouterModel(apiKey);
  
  for (const lang of testLanguages) {
    console.log(`\n🌐 測試語言: ${lang}`);
    
    try {
      // 創建 OverviewSummary 實例
      const overviewSummary = new OverviewSummary(
        mockData,
        model,
        "Test context",
        lang
      );
      
      // 測試 oneShotSummary 方法
      console.log(`   📝 測試 oneShotSummary...`);
      const startTime = Date.now();
      const oneShotResult = await overviewSummary.oneShotSummary();
      const duration = Date.now() - startTime;
      
      console.log(`   ✅ oneShotSummary 完成，耗時: ${duration}ms`);
      console.log(`   📊 結果長度: ${oneShotResult.length} 字符`);
      console.log(`   📝 結果預覽: ${oneShotResult.substring(0, 200)}...`);
      
      // 檢查結果是否包含對應語言的內容
      if (lang === "zh-TW" || lang === "zh-CN") {
        if (oneShotResult.includes("主題") || oneShotResult.includes("主题")) {
          console.log(`   ✓ ${lang}: 結果包含中文內容`);
        } else {
          console.log(`   ⚠️ ${lang}: 結果可能缺少中文內容`);
        }
      } else if (lang === "en") {
        if (oneShotResult.includes("Topic")) {
          console.log(`   ✓ ${lang}: 結果包含英文內容`);
        } else {
          console.log(`   ⚠️ ${lang}: 結果可能缺少英文內容`);
        }
      }
      
      // 檢查結果格式
      if (oneShotResult.includes("* **") || oneShotResult.includes("* __")) {
        console.log(`   ✓ ${lang}: 結果格式正確（包含 markdown 列表）`);
      } else {
        console.log(`   ⚠️ ${lang}: 結果格式可能不正確`);
      }
      
    } catch (error) {
      console.error(`   ❌ ${lang}: 測試失敗:`, error);
    }
  }
}

// 測試錯誤處理和邊界情況
async function testErrorHandling() {
  console.log("\n🧪 開始測試錯誤處理和邊界情況...");
  
  try {
    // 測試無效語言
    console.log("📝 測試無效語言處理...");
    const invalidLang = "invalid-lang" as SupportedLanguage;
    const prompt = getOverviewOneShotPrompt(invalidLang, ["Test Topic"]);
    
    // 應該回退到英文
    if (prompt.includes("Your job")) {
      console.log("✅ 無效語言正確回退到英文");
    } else {
      console.log("❌ 無效語言處理失敗");
    }
    
    // 測試空主題列表
    console.log("📝 測試空主題列表處理...");
    const emptyTopicsPrompt = getOverviewOneShotPrompt("en", []);
    if (emptyTopicsPrompt.includes("Here are the topics:")) {
      console.log("✅ 空主題列表處理正確");
    } else {
      console.log("❌ 空主題列表處理失敗");
    }
    
    // 測試空主題名稱
    console.log("📝 測試空主題名稱處理...");
    const emptyTopicPrompt = getOverviewPerTopicPrompt("en", "");
    if (emptyTopicPrompt.includes("following topic:")) {
      console.log("✅ 空主題名稱處理正確");
    } else {
      console.log("❌ 空主題名稱處理失敗");
    }
    
  } catch (error) {
    console.error("❌ 錯誤處理測試失敗:", error);
  }
}

// 測試 prompt 內容的一致性
async function testPromptConsistency() {
  console.log("\n🧪 開始測試 prompt 內容的一致性...");
  
  const testLanguages: SupportedLanguage[] = ["en", "zh-TW", "zh-CN", "fr", "es", "ja"];
  const testTopicNames = ["專制很好 (30%)", "民主很好 (70%)"];
  
  for (const lang of testLanguages) {
    console.log(`📝 檢查語言 ${lang} 的 prompt 一致性...`);
    
    try {
      const prompt = getOverviewOneShotPrompt(lang, testTopicNames);
      
      // 檢查是否包含必要的結構元素
      if (prompt.includes("markdown") || prompt.includes("列表") || prompt.includes("liste") || prompt.includes("lista") || prompt.includes("リスト")) {
        console.log(`   ✓ ${lang}: 包含輸出格式說明`);
      } else {
        console.log(`   ⚠️ ${lang}: 可能缺少輸出格式說明`);
      }
      
      if (prompt.includes("statements") || prompt.includes("陳述") || prompt.includes("陈述") || prompt.includes("déclarations") || prompt.includes("declaraciones") || prompt.includes("声明")) {
        console.log(`   ✓ ${lang}: 包含術語說明`);
      } else {
        console.log(`   ⚠️ ${lang}: 可能缺少術語說明`);
      }
      
      if (prompt.includes("participant") || prompt.includes("參與者") || prompt.includes("参与者") || prompt.includes("participant") || prompt.includes("participante") || prompt.includes("参加者")) {
        console.log(`   ✓ ${lang}: 包含參與者說明`);
      } else {
        console.log(`   ⚠️ ${lang}: 可能缺少參與者說明`);
      }
      
    } catch (error) {
      console.error(`   ❌ ${lang}: 一致性檢查失敗:`, error);
    }
  }
}

// 測試真實 LLM 回應的語言檢測
async function testLLMLanguageDetection() {
  console.log("\n🧪 開始測試 LLM 回應的語言檢測...");
  
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    console.log("⚠️ 未設定 OPENROUTER_API_KEY 環境變數，跳過 LLM 語言檢測測試");
    return;
  }
  
  const model = new OpenRouterModel(apiKey);
  const testLanguages: SupportedLanguage[] = ["en", "zh-TW", "zh-CN", "fr", "es", "ja"];
  const testTopicNames = ["專制很好 (30%)", "民主很好 (70%)"];
  
  for (const lang of testLanguages) {
    console.log(`\n🌐 測試 LLM 語言: ${lang}`);
    
    try {
      // 生成 prompt
      const prompt = getOverviewOneShotPrompt(lang, testTopicNames);
      console.log(`   📝 生成 ${lang} 語言的 prompt`);
      
      // 調用 LLM
      console.log(`   🤖 調用 LLM 生成回應...`);
      const startTime = Date.now();
      const response = await model.generateText(prompt, lang);
      const duration = Date.now() - startTime;
      
      console.log(`   ✅ LLM 回應完成，耗時: ${duration}ms`);
      console.log(`   📊 回應長度: ${response.length} 字符`);
      console.log(`   📝 回應預覽: ${response.substring(0, 200)}...`);
      
      // 檢測回應語言
      const languageIndicators: Record<SupportedLanguage, string[]> = {
        "en": ["Topic", "summary", "discussion", "statements", "participants", "analysis", "findings", "conclusions", "recommendations", "insights", "perspectives", "views", "opinions", "arguments", "evidence", "data", "research", "study", "report", "overview"],
        "zh-TW": ["主題", "摘要", "討論", "陳述", "參與者", "分析", "發現", "結論", "建議", "見解", "觀點", "意見", "論點", "證據", "資料", "研究", "報告", "概觀", "議題", "內容", "重點", "核心", "關鍵", "主要", "重要", "相關", "影響", "結果", "趨勢", "方向"],
        "zh-CN": ["主题", "摘要", "讨论", "陈述", "参与者", "分析", "发现", "结论", "建议", "见解", "观点", "意见", "论点", "证据", "资料", "研究", "报告", "概观", "议题", "内容", "重点", "核心", "关键", "主要", "重要", "相关", "影响", "结果", "趋势", "方向"],
        "fr": ["Sujet", "résumé", "discussion", "déclarations", "participants", "analyse", "découvertes", "conclusions", "recommandations", "perspectives", "points de vue", "opinions", "arguments", "preuves", "données", "recherche", "étude", "rapport", "aperçu", "thème", "contenu", "points clés", "noyau", "essentiel", "principal", "important", "pertinent", "impact", "résultats", "tendances"],
        "es": ["Tema", "resumen", "discusión", "declaraciones", "participantes", "análisis", "descubrimientos", "conclusiones", "recomendaciones", "perspectivas", "puntos de vista", "opiniones", "argumentos", "evidencias", "datos", "investigación", "estudio", "informe", "descripción general", "contenido", "puntos clave", "núcleo", "esencial", "principal", "importante", "relevante", "impacto", "resultados", "tendencias"],
        "ja": ["トピック", "要約", "議論", "声明", "参加者", "分析", "発見", "結論", "推奨", "洞察", "視点", "意見", "議論", "証拠", "データ", "研究", "調査", "報告", "概要", "内容", "要点", "核心", "重要", "主要", "関連", "影響", "結果", "傾向", "方向性"]
      };
      
      const indicators = languageIndicators[lang];
      if (indicators) {
        // 使用更靈活的匹配邏輯
        let matchCount = 0;
        let totalScore = 0;
        
        for (const indicator of indicators) {
          if (response.toLowerCase().includes(indicator.toLowerCase())) {
            matchCount++;
            totalScore += 1;
          }
        }
        
        // 額外的語言特定檢測
        if (lang === "zh-TW" || lang === "zh-CN") {
          // 檢測中文字符
          const chineseCharCount = (response.match(/[\u4e00-\u9fff]/g) || []).length;
          if (chineseCharCount > 10) {
            totalScore += 2; // 中文字符給予額外分數
          }
        } else if (lang === "ja") {
          // 檢測日文字符
          const japaneseCharCount = (response.match(/[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]/g) || []).length;
          if (japaneseCharCount > 5) {
            totalScore += 2;
          }
        } else if (lang === "fr") {
          // 檢測法文特徵
          const frenchFeatures = response.match(/[àâäéèêëïîôöùûüÿç]/gi);
          if (frenchFeatures && frenchFeatures.length > 2) {
            totalScore += 1;
          }
        } else if (lang === "es") {
          // 檢測西班牙文特徵
          const spanishFeatures = response.match(/[áéíóúñü]/gi);
          if (spanishFeatures && spanishFeatures.length > 2) {
            totalScore += 1;
          }
        }
        
        const matchRate = (totalScore / (indicators.length + 2)) * 100; // +2 為額外檢測項目
        
        if (matchRate >= 30) {
          console.log(`   ✓ ${lang}: 回應語言檢測成功 (匹配率: ${matchRate.toFixed(1)}%)`);
        } else if (matchRate >= 15) {
          console.log(`   ⚠️ ${lang}: 回應語言檢測部分成功 (匹配率: ${matchRate.toFixed(1)}%)`);
        } else {
          console.log(`   ❌ ${lang}: 回應語言檢測失敗 (匹配率: ${matchRate.toFixed(1)}%)`);
        }
        
        // 顯示詳細的匹配信息
        console.log(`   📊 關鍵詞匹配: ${matchCount}/${indicators.length}`);
        if (lang === "zh-TW" || lang === "zh-CN") {
          const chineseCharCount = (response.match(/[\u4e00-\u9fff]/g) || []).length;
          console.log(`   📝 中文字符數量: ${chineseCharCount}`);
        }
      }
      
    } catch (error) {
      console.error(`   ❌ ${lang}: LLM 語言檢測測試失敗:`, error);
    }
  }
}

// 主測試函數
async function runAllTests() {
  console.log("🚀 開始執行 Overview.ts 多語言功能測試套件...\n");
  
  try {
    await testMultilangPromptGeneration();
    await testOverviewSummaryMultilang();
    await testErrorHandling();
    await testPromptConsistency();
    await testLLMLanguageDetection();
    
    console.log("\n🎉 所有測試完成！");
    
  } catch (error) {
    console.error("\n💥 測試執行過程中發生錯誤:", error);
  }
}

// 執行測試
if (require.main === module) {
  runAllTests().catch(console.error);
}

export {
  testMultilangPromptGeneration,
  testOverviewSummaryMultilang,
  testErrorHandling,
  testPromptConsistency,
  testLLMLanguageDetection,
  runAllTests
};
