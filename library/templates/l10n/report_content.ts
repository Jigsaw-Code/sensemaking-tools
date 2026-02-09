import { SupportedLanguage } from "./languages";

// Define the structure for content sections
type ContentSection = {
  [key: string]: string;
};

type ContentStructure = {
  [section: string]: {
    [lang in SupportedLanguage]: ContentSection;
  };
};

// Report content and descriptions
export const REPORT_CONTENT: ContentStructure = {
  introduction: {
    "en": {
      text: "This report summarizes the results of public input, encompassing:",
      statements: "statements",
      votes: "votes",
      topics: "topics",
      subtopics: "subtopics",
      anonymous: "All voters were anonymous."
    },
    "zh-TW": {
      text: "本報告總結了公眾意見的結果，包含：",
      statements: "個意見",
      votes: "個投票",
      topics: "個主題",
      subtopics: "個子主題",
      anonymous: "所有投票者都是匿名的。"
    },
    "fr": {
      text: "Ce rapport résume les résultats de la contribution publique, comprenant :",
      statements: "déclarations",
      votes: "votes",
      topics: "sujets",
      subtopics: "sous-sujets",
      anonymous: "Tous les électeurs étaient anonymes."
    },
    "zh-CN": {
      text: "本报告总结了公众意见的结果，包含：",
      statements: "个意见",
      votes: "个投票",
      topics: "个主题",
      subtopics: "个子主题",
      anonymous: "所有投票者都是匿名的。"
    },
    "es": {
      text: "Este informe resume los resultados de la contribución pública, incluyendo:",
      statements: "declaraciones",
      votes: "votos",
      topics: "temas",
      subtopics: "subtemas",
      anonymous: "Todos los votantes eran anónimos."
    },
    "ja": {
      text: "このレポートは、以下の内容を含む公的意見の結果をまとめたものです：",
      statements: "声明",
      votes: "投票",
      topics: "トピック",
      subtopics: "サブトピック",
      anonymous: "すべての投票者は匿名でした。"
    }
  },
  overview: {
    "en": {
      preamble: "Below is a high level overview of the topics discussed in the conversation, as well as the percentage of statements categorized under each topic. Note that the percentages may add up to greater than 100% when statements fall under more than one topic.\n\n"
    },
    "zh-TW": {
      preamble: "以下是對話中討論主題的高層次概述，以及每個主題下分類的意見百分比。請注意，當意見屬於多個主題時，百分比總和可能超過 100%。\n\n"
    },
    "fr": {
      preamble: "Voici un aperçu de haut niveau des sujets discutés dans la conversation, ainsi que le pourcentage de déclarations classées sous chaque sujet. Notez que les pourcentages peuvent s'ajouter à plus de 100% lorsque les déclarations relèvent de plusieurs sujets.\n\n"
    },
    "zh-CN": {
      preamble: "以下是对话中讨论主题的高层次概述，以及每个主题下分类的意见百分比。请注意，当意见属于多个主题时，百分比总和可能超过 100%。\n\n"
    },
    "es": {
      preamble: "A continuación se presenta una visión general de alto nivel de los temas discutidos en la conversación, así como el porcentaje de declaraciones categorizadas bajo cada tema. Tenga en cuenta que los porcentajes pueden sumar más del 100% cuando las declaraciones caen bajo más de un tema.\n\n"
    },
    "ja": {
      preamble: "以下は、会話で議論されたトピックの高レベルな概要と、各トピックの下に分類された声明の割合です。声明が複数のトピックに該当する場合、割合の合計が100%を超える可能性があることにご注意ください。\n\n"
    }
  },
  topics: {
    "en": {
      overview: "From the statements submitted, {topicCount} high level topics were identified{subtopicsText}. Based on voting patterns{groupsText} both points of common ground as well as differences of opinion {groupsBetweenText}have been identified and are described below.\n\n"
    },
    "zh-TW": {
      overview: "從提交的意見中，識別出 {topicCount} 個高層次主題{subtopicsText}。基於投票模式{groupsText} 已識別出共同點以及意見分歧 {groupsBetweenText}，並在下面描述。\n\n"
    },
    "fr": {
      overview: "À partir des déclarations soumises, {topicCount} sujets de haut niveau ont été identifiés{subtopicsText}. Sur la base des modèles de vote{groupsText} à la fois les points de terrain d'entente ainsi que les différences d'opinion {groupsBetweenText}ont été identifiés et sont décrits ci-dessous.\n\n"
    },
    "zh-CN": {
      overview: "从提交的意见中，识别出 {topicCount} 个高层次主题{subtopicsText}。基于投票模式{groupsText} 已识别出共同点以及意见分歧 {groupsBetweenText}，并在下面描述。\n\n"
    },
    "es": {
      overview: "De las declaraciones presentadas, se identificaron {topicCount} temas de alto nivel{subtopicsText}. Basándose en los patrones de votación{groupsText} tanto los puntos de terreno común como las diferencias de opinión {groupsBetweenText}han sido identificados y se describen a continuación.\n\n"
    },
    "ja": {
      overview: "提出された声明から、{topicCount}の高レベルトピック{subtopicsText}が特定されました。投票パターンに基づいて{groupsText}、共通点と意見の相違点の両方{groupsBetweenText}が特定され、以下に説明されています。\n\n"
    }
  },
  subtopics: {
    "en": {
      text: "as well as {count} subtopics"
    },
    "zh-TW": {
      text: "，以及 {count} 個子主題"
    },
    "fr": {
      text: ", ainsi que {count} sous-sujets"
    },
    "zh-CN": {
      text: "，以及 {count} 个子主题"
    },
    "es": {
      text: ", así como {count} subtemas"
    },
    "ja": {
      text: "、および {count} のサブトピック"
    }
  },
  topSubtopics: {
    "en": {
      text: "{totalCount} subtopics of discussion emerged. These {topCount} subtopics had the most statements submitted."
    },
    "zh-TW": {
      text: "討論中出現了 {totalCount} 個子主題。這 {topCount} 個子主題收到的意見最多。"
    },
    "fr": {
      text: "{totalCount} sous-sujets de discussion ont émergé. Ces {topCount} sous-sujets avaient le plus de déclarations soumises."
    },
    "zh-CN": {
      text: "讨论中出现了 {totalCount} 个子主题。这 {topCount} 个子主题收到的意见最多。"
    },
    "es": {
      text: "Emergieron {totalCount} subtemas de discusión. Estos {topCount} subtemas tuvieron la mayor cantidad de declaraciones presentadas."
    },
    "ja": {
      text: "議論から{totalCount}のサブトピックが生まれました。これらの{topCount}のサブトピックには、最も多くの声明が提出されました。"
    }
  },
  opinionGroups: {
    "en": {
      text: "{groupCount} distinct groups (named here as {groupNames}) emerged with differing viewpoints in relation to the submitted statements. The groups are based on people who tend to vote more similarly to each other than to those outside the group. However there are points of common ground where the groups voted similarly.\n\n"
    },
    "zh-TW": {
      text: "{groupCount} 個不同的群組（這裡命名為 {groupNames}）在提交的意見方面出現了不同的觀點。這些群組基於傾向於彼此投票更相似的人，而不是與群組外的人投票相似。然而，在群組投票相似的地方存在共同點。\n\n"
    },
    "fr": {
      text: "{groupCount} groupes distincts (nommés ici {groupNames}) ont émergé avec des points de vue différents par rapport aux déclarations soumises. Les groupes sont basés sur des personnes qui ont tendance à voter plus similairement les uns aux autres qu'à ceux en dehors du groupe. Cependant, il y a des points de terrain d'entente où les groupes ont voté de manière similaire.\n\n"
    },
    "zh-CN": {
      text: "{groupCount} 个不同的群组（这里命名为 {groupNames}）在提交的意见方面出现了不同的观点。这些群组基于倾向于彼此投票更相似的人，而不是与群组外的人投票相似。然而，在群组投票相似的地方存在共同点。\n\n"
    },
    "es": {
      text: "{groupCount} grupos distintos (nombrados aquí como {groupNames}) surgieron con puntos de vista diferentes en relación con las declaraciones presentadas. Los grupos se basan en personas que tienden a votar de manera más similar entre sí que con aquellos fuera del grupo. Sin embargo, hay puntos de terreno común donde los grupos votaron de manera similar.\n\n"
    },
    "ja": {
      text: "{groupCount}の異なるグループ（ここでは{groupNames}として命名）が、提出された声明に関連して異なる視点で出現しました。これらのグループは、グループ外の人よりも互いに似た投票をする傾向がある人々に基づいています。しかし、グループが同様に投票した共通点が存在します。\n\n"
    }
  }
};

export function getReportContent(
  section: keyof typeof REPORT_CONTENT,
  subsection: string,
  lang: SupportedLanguage,
  replacements?: Record<string, string | number>
): string {
  const content = REPORT_CONTENT[section][lang] || REPORT_CONTENT[section]["en"];
  let text = content[subsection] as string;
  
  if (replacements) {
    Object.entries(replacements).forEach(([key, value]) => {
      text = text.replace(new RegExp(`{${key}}`, 'g'), value.toString());
    });
  }
  
  return text;
}
