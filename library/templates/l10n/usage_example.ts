// Example of how to use the localization system in existing summary classes
import { 
  getReportSectionTitle, 
  getReportContent, 
  getSubsectionTitle,
  getTopicSummaryText,
  getPluralForm,
  type SupportedLanguage 
} from './index';

// Example: Refactoring IntroSummary class
export class IntroSummaryExample {
  private output_lang: SupportedLanguage;

  constructor(output_lang: SupportedLanguage = "en") {
    this.output_lang = output_lang;
  }

  async getSummary() {
    const lang = this.output_lang;
    
    // Get section title
    const title = getReportSectionTitle("introduction", lang);
    
    // Get content with dynamic values
    const text = getReportContent("introduction", "text", lang);
    const statementsLabel = getReportContent("introduction", "statements", lang);
    const votesLabel = getReportContent("introduction", "votes", lang);
    const topicsLabel = getReportContent("introduction", "topics", lang);
    const subtopicsLabel = getReportContent("introduction", "subtopics", lang);
    const anonymousText = getReportContent("introduction", "anonymous", lang);
    
    // Build the content
    const content = `${text}\n` +
      ` * __{commentCount} ${statementsLabel}__\n` +
      ` * __{voteCount} ${votesLabel}__\n` +
      ` * {topicCount} ${topicsLabel}\n` +
      ` * {subtopicCount} ${subtopicsLabel}\n\n` +
      `${anonymousText}`;
    
    return { title, text: content };
  }
}

// Example: Refactoring OverviewSummary class
export class OverviewSummaryExample {
  private output_lang: SupportedLanguage;

  constructor(output_lang: SupportedLanguage = "en") {
    this.output_lang = output_lang;
  }

  async getSummary() {
    const lang = this.output_lang;
    
    // Get section title
    const title = getReportSectionTitle("overview", lang);
    
    // Get preamble text
    const preamble = getReportContent("overview", "preamble", lang);
    
    // Get the summary content (this would come from LLM)
    const result = await this.generateSummaryContent();
    
    return { title, text: preamble + result };
  }

  private async generateSummaryContent() {
    // This would call the LLM with English prompts
    // The language prefix ensures output is in the target language
    return "Generated content...";
  }
}

// Example: Refactoring TopicsSummary class
export class TopicsSummaryExample {
  private output_lang: SupportedLanguage;

  constructor(output_lang: SupportedLanguage = "en") {
    this.output_lang = output_lang;
  }

  async getSummary() {
    const lang = this.output_lang;
    
    // Get section title
    const title = getReportSectionTitle("topics", lang);
    
    // Get overview text with replacements
    const overviewText = getReportContent("topics", "overview", lang, {
      topicCount: 5,
      subtopicsText: ", as well as 12 subtopics",
      groupsText: " between the opinion groups described above,",
      groupsBetweenText: "between the groups "
    });
    
    return { title, text: overviewText };
  }
}

// Example: Refactoring TopSubtopicsSummary class
export class TopSubtopicsSummaryExample {
  private output_lang: SupportedLanguage;

  constructor(output_lang: SupportedLanguage = "en") {
    this.output_lang = output_lang;
  }

  async getSummary() {
    const lang = this.output_lang;
    const topCount = 5;
    const totalCount = 20;
    
    // Get section title with count
    const title = getReportSectionTitle("topSubtopics", lang, topCount);
    
    // Get content text with replacements
    const text = getReportContent("topSubtopics", "text", lang, {
      totalCount,
      topCount
    });
    
    return { title, text };
  }
}

// Example: Refactoring OpinionGroupsSummary class
export class OpinionGroupsSummaryExample {
  private output_lang: SupportedLanguage;

  constructor(output_lang: SupportedLanguage = "en") {
    this.output_lang = output_lang;
  }

  async getSummary() {
    const lang = this.output_lang;
    const groupCount = 3;
    const groupNames = '"Group A", "Group B", "Group C"';
    
    // Get section title
    const title = getReportSectionTitle("opinionGroups", lang);
    
    // Get content text with replacements
    const text = getReportContent("opinionGroups", "text", lang, {
      groupCount,
      groupNames
    });
    
    return { title, text };
  }
}

// Example: Refactoring subsection titles
export class SubsectionTitlesExample {
  private output_lang: SupportedLanguage;

  constructor(output_lang: SupportedLanguage = "en") {
    this.output_lang = output_lang;
  }

  getThemesTitle() {
    return getSubsectionTitle("prominentThemes", this.output_lang);
  }

  getCommonGroundTitle(hasGroups: boolean) {
    const section = hasGroups ? "commonGroundBetweenGroups" : "commonGround";
    return getSubsectionTitle(section, this.output_lang);
  }

  getDifferencesTitle() {
    return getSubsectionTitle("differencesOfOpinion", this.output_lang);
  }

  getOtherStatementsTitle(count: number) {
    return getSubsectionTitle("otherStatements", this.output_lang, count);
  }
}

// Example: Refactoring topic summaries
export class TopicSummariesExample {
  private output_lang: SupportedLanguage;

  constructor(output_lang: SupportedLanguage = "en") {
    this.output_lang = output_lang;
  }

  getTopicSummaryText(subtopicCount: number, statementCount: number) {
    const lang = this.output_lang;
    
    return getTopicSummaryText("topicSummary", lang, {
      subtopicCount,
      subtopicPlural: getPluralForm(subtopicCount, lang),
      statementCount,
      statementPlural: getPluralForm(statementCount, lang)
    });
  }

  getRelativeAgreementText(level: string) {
    const lang = this.output_lang;
    
    return getTopicSummaryText("relativeAgreement", lang, {
      level
    });
  }
}
