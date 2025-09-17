// Test file for the localization system
import { 
  getReportSectionTitle, 
  getReportContent, 
  getSubsectionTitle,
  getTopicSummaryText,
  getPluralForm,
  getLanguagePrefix,
  isValidLanguage,
  getStatisticsMessage
} from './index';

// Test function
function testLocalization() {
  console.log("🧪 Testing Localization System...\n");

  // Test 1: Language validation
  console.log("1. Testing language validation:");
  console.log(`   "en" is valid: ${isValidLanguage("en")}`);
  console.log(`   "zh-TW" is valid: ${isValidLanguage("zh-TW")}`);
  console.log(`   "zh-CN" is valid: ${isValidLanguage("zh-CN")}`);
  console.log(`   "fr" is valid: ${isValidLanguage("fr")}`);
  console.log(`   "es" is valid: ${isValidLanguage("es")}`);
  console.log(`   "ja" is valid: ${isValidLanguage("ja")}`);
  console.log(`   "invalid" is valid: ${isValidLanguage("invalid")}\n`);

  // Test 2: Language prefixes
  console.log("2. Testing language prefixes:");
  console.log(`   English prefix: "${getLanguagePrefix("en")}"`);
  console.log(`   Traditional Chinese prefix: "${getLanguagePrefix("zh-TW")}"`);
  console.log(`   Simplified Chinese prefix: "${getLanguagePrefix("zh-CN")}"`);
  console.log(`   French prefix: "${getLanguagePrefix("fr")}"`);
  console.log(`   Spanish prefix: "${getLanguagePrefix("es")}"`);
  console.log(`   Japanese prefix: "${getLanguagePrefix("ja")}"\n`);

  // Test 3: Report section titles
  console.log("3. Testing report section titles:");
  console.log(`   Introduction (en): ${getReportSectionTitle("introduction", "en")}`);
  console.log(`   Introduction (zh-TW): ${getReportSectionTitle("introduction", "zh-TW")}`);
  console.log(`   Introduction (zh-CN): ${getReportSectionTitle("introduction", "zh-CN")}`);
  console.log(`   Introduction (fr): ${getReportSectionTitle("introduction", "fr")}`);
  console.log(`   Introduction (es): ${getReportSectionTitle("introduction", "es")}`);
  console.log(`   Introduction (ja): ${getReportSectionTitle("introduction", "ja")}`);
  console.log(`   Top Subtopics (en, count=5): ${getReportSectionTitle("topSubtopics", "en", 5)}`);
  console.log(`   Top Subtopics (zh-TW, count=5): ${getReportSectionTitle("topSubtopics", "zh-TW", 5)}`);
  console.log(`   Top Subtopics (zh-CN, count=5): ${getReportSectionTitle("topSubtopics", "zh-CN", 5)}`);
  console.log(`   Top Subtopics (fr, count=5): ${getReportSectionTitle("topSubtopics", "fr", 5)}`);
  console.log(`   Top Subtopics (es, count=5): ${getReportSectionTitle("topSubtopics", "es", 5)}`);
  console.log(`   Top Subtopics (ja, count=5): ${getReportSectionTitle("topSubtopics", "ja", 5)}\n`);

  // Test 4: Report content
  console.log("4. Testing report content:");
  console.log(`   Topics overview (en): ${getReportContent("topics", "overview", "en", {
    topicCount: 3,
    subtopicsText: ", as well as 8 subtopics",
    groupsText: " between opinion groups,",
    groupsBetweenText: "between groups "
  })}`);
  console.log(`   Topics overview (zh-TW): ${getReportContent("topics", "overview", "zh-TW", {
    topicCount: 3,
    subtopicsText: ", as well as 8 subtopics",
    groupsText: " between opinion groups,",
    groupsBetweenText: "between groups "
  })}`);
  console.log(`   Topics overview (zh-CN): ${getReportContent("topics", "overview", "zh-CN", {
    topicCount: 3,
    subtopicsText: ", as well as 8 subtopics",
    groupsText: " between opinion groups,",
    groupsBetweenText: "between groups "
  })}`);
  console.log(`   Topics overview (fr): ${getReportContent("topics", "overview", "fr", {
    topicCount: 3,
    subtopicsText: ", as well as 8 subtopics",
    groupsText: " between opinion groups,",
    groupsBetweenText: "between groups "
  })}`);
  console.log(`   Topics overview (es): ${getReportContent("topics", "overview", "es", {
    topicCount: 3,
    subtopicsText: ", as well as 8 subtopics",
    groupsText: " between opinion groups,",
    groupsBetweenText: "between groups "
  })}`);
  console.log(`   Topics overview (ja): ${getReportContent("topics", "overview", "ja", {
    topicCount: 3,
    subtopicsText: ", as well as 8 subtopics",
    groupsText: " between opinion groups,",
    groupsBetweenText: "between groups "
  })}\n`);

  // Test 5: Subsection titles
  console.log("5. Testing subsection titles:");
  console.log(`   Prominent themes (en): ${getSubsectionTitle("prominentThemes", "en")}`);
  console.log(`   Prominent themes (zh-TW): ${getSubsectionTitle("prominentThemes", "zh-TW")}`);
  console.log(`   Prominent themes (zh-CN): ${getSubsectionTitle("prominentThemes", "zh-CN")}`);
  console.log(`   Prominent themes (fr): ${getSubsectionTitle("prominentThemes", "fr")}`);
  console.log(`   Prominent themes (es): ${getSubsectionTitle("prominentThemes", "es")}`);
  console.log(`   Prominent themes (ja): ${getSubsectionTitle("prominentThemes", "ja")}`);
  console.log(`   Common ground (en): ${getSubsectionTitle("commonGround", "en")}`);
  console.log(`   Common ground (zh-TW): ${getSubsectionTitle("commonGround", "zh-TW")}`);
  console.log(`   Common ground (zh-CN): ${getSubsectionTitle("commonGround", "zh-CN")}`);
  console.log(`   Common ground (fr): ${getSubsectionTitle("commonGround", "fr")}`);
  console.log(`   Common ground (es): ${getSubsectionTitle("commonGround", "es")}`);
  console.log(`   Common ground (ja): ${getSubsectionTitle("commonGround", "ja")}\n`);

  // Test 6: Topic summaries
  console.log("6. Testing topic summaries:");
  console.log(`   Topic summary (en): ${getTopicSummaryText("topicSummary", "en", {
    subtopicCount: 3,
    subtopicPlural: getPluralForm(3, "en"),
    statementCount: 15,
    statementPlural: getPluralForm(15, "en")
  })}`);
  console.log(`   Topic summary (zh-TW): ${getTopicSummaryText("topicSummary", "zh-TW", {
    subtopicCount: 3,
    subtopicPlural: getPluralForm(3, "zh-TW"),
    statementCount: 15,
    statementPlural: getPluralForm(15, "zh-TW")
  })}`);
  console.log(`   Topic summary (zh-CN): ${getTopicSummaryText("topicSummary", "zh-CN", {
    subtopicCount: 3,
    subtopicPlural: getPluralForm(3, "zh-CN"),
    statementCount: 15,
    statementPlural: getPluralForm(15, "zh-CN")
  })}`);
  console.log(`   Topic summary (fr): ${getTopicSummaryText("topicSummary", "fr", {
    subtopicCount: 3,
    subtopicPlural: getPluralForm(3, "fr"),
    statementCount: 15,
    statementPlural: getPluralForm(15, "fr")
  })}`);
  console.log(`   Topic summary (es): ${getTopicSummaryText("topicSummary", "es", {
    subtopicCount: 3,
    subtopicPlural: getPluralForm(3, "es"),
    statementCount: 15,
    statementPlural: getPluralForm(15, "es")
  })}`);
  console.log(`   Topic summary (ja): ${getTopicSummaryText("topicSummary", "ja", {
    subtopicCount: 3,
    subtopicPlural: getPluralForm(3, "ja"),
    statementCount: 15,
    statementPlural: getPluralForm(15, "ja")
  })}\n`);

  // Test 7: Plural forms
  console.log("7. Testing plural forms:");
  console.log(`   English: 1 subtopic${getPluralForm(1, "en")}, 2 subtopic${getPluralForm(2, "en")}`);
  console.log(`   Traditional Chinese: 1 個子主題${getPluralForm(1, "zh-TW")}, 2 個子主題${getPluralForm(2, "zh-TW")}`);
  console.log(`   Simplified Chinese: 1 个子主题${getPluralForm(1, "zh-CN")}, 2 个子主题${getPluralForm(2, "zh-CN")}`);
  console.log(`   French: 1 sous-sujet${getPluralForm(1, "fr")}, 2 sous-sujet${getPluralForm(2, "fr")}`);
  console.log(`   Spanish: 1 subtema${getPluralForm(1, "es")}, 2 subtema${getPluralForm(2, "es")}`);
  console.log(`   Japanese: 1 サブトピック${getPluralForm(1, "ja")}, 2 サブトピック${getPluralForm(2, "ja")}\n`);

  // Test 8: Statistics messages (new statements functionality)
  console.log("8. Testing statistics messages:");
  console.log(`   Statements (en): ${getStatisticsMessage("statements", "en", {})}`);
  console.log(`   Statements (zh-TW): ${getStatisticsMessage("statements", "zh-TW", {})}`);
  console.log(`   Statements (zh-CN): ${getStatisticsMessage("statements", "zh-CN", {})}`);
  console.log(`   Statements (fr): ${getStatisticsMessage("statements", "fr", {})}`);
  console.log(`   Statements (es): ${getStatisticsMessage("statements", "es", {})}`);
  console.log(`   Statements (ja): ${getStatisticsMessage("statements", "ja", {})}\n`);

  console.log("✅ Localization system test completed!");
}

// Run the test if this file is executed directly
if (require.main === module) {
  testLocalization();
}

export { testLocalization };
